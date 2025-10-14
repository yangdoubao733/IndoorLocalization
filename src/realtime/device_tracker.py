"""
设备跟踪模块
管理多个非合作目标设备的实时定位
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading


@dataclass
class TargetDevice:
    """目标设备信息"""
    mac: str                                    # 设备标识（MAC地址/IMEI/RFID标签ID等）
    name: str = ""                              # 设备名称（可选）
    signal_type: str = "WiFi"                   # 信号类型（WiFi/蓝牙/手机信号等）
    last_seen: datetime = field(default_factory=datetime.now)  # 最后检测时间
    position: Optional[np.ndarray] = None       # 当前位置 (x,y,z)
    rssi: Optional[np.ndarray] = None           # 当前RSSI值
    confidence: float = 0.0                     # 定位置信度
    trajectory: List[Tuple[datetime, np.ndarray]] = field(default_factory=list)  # 轨迹历史
    frequency: Optional[float] = None           # 工作频率（Hz）
    tx_power: Optional[float] = None            # 发射功率（dBm）

    def update(self, position: np.ndarray, rssi: np.ndarray, confidence: float):
        """更新设备状态"""
        self.position = position
        self.rssi = rssi
        self.confidence = confidence
        self.last_seen = datetime.now()

        # 保存轨迹（限制最多100个点）
        self.trajectory.append((self.last_seen, position.copy()))
        if len(self.trajectory) > 100:
            self.trajectory.pop(0)

    def get_trajectory_array(self) -> np.ndarray:
        """获取轨迹数组"""
        if not self.trajectory:
            return np.array([])
        return np.array([pos for _, pos in self.trajectory])

    def is_active(self, timeout_seconds: float = 10.0) -> bool:
        """判断设备是否活跃"""
        elapsed = (datetime.now() - self.last_seen).total_seconds()
        return elapsed < timeout_seconds


class DeviceTracker:
    """设备跟踪器"""

    def __init__(self, signal_collector, localization_engine,
                 update_interval: float = 1.0,
                 device_timeout: float = 30.0):
        """
        初始化

        Args:
            signal_collector: 信号采集器
            localization_engine: 定位引擎
            update_interval: 更新间隔（秒）
            device_timeout: 设备超时时间（秒）
        """
        self.signal_collector = signal_collector
        self.localization_engine = localization_engine
        self.update_interval = update_interval
        self.device_timeout = device_timeout

        # 设备字典 MAC -> TargetDevice
        self.devices: Dict[str, TargetDevice] = {}

        # 跟踪控制
        self.is_tracking = False
        self.tracking_thread = None
        self.lock = threading.Lock()

        print(f"设备跟踪器初始化完成")
        print(f"  更新间隔: {update_interval}秒")
        print(f"  设备超时: {device_timeout}秒")

    def add_device(self, mac: str, name: str = "", signal_type: str = "WiFi",
                   frequency: Optional[float] = None, tx_power: Optional[float] = None):
        """
        添加要跟踪的设备

        Args:
            mac: 设备标识（MAC地址/IMEI/RFID等）
            name: 设备名称（可选）
            signal_type: 信号类型（WiFi/Bluetooth/Cellular等）
            frequency: 工作频率（Hz，可选）
            tx_power: 发射功率（dBm，可选）
        """
        with self.lock:
            if mac not in self.devices:
                self.devices[mac] = TargetDevice(
                    mac=mac,
                    name=name,
                    signal_type=signal_type,
                    frequency=frequency,
                    tx_power=tx_power
                )
                print(f"添加跟踪设备: {mac} ({name}) - {signal_type}")

    def remove_device(self, mac: str):
        """移除设备"""
        with self.lock:
            if mac in self.devices:
                del self.devices[mac]
                print(f"移除设备: {mac}")

    def get_device(self, mac: str) -> Optional[TargetDevice]:
        """获取设备信息"""
        with self.lock:
            return self.devices.get(mac)

    def get_all_devices(self) -> List[TargetDevice]:
        """获取所有设备"""
        with self.lock:
            return list(self.devices.values())

    def get_active_devices(self) -> List[TargetDevice]:
        """获取活跃设备"""
        with self.lock:
            return [dev for dev in self.devices.values()
                   if dev.is_active(self.device_timeout)]

    def update_device_location(self, mac: str) -> bool:
        """
        更新单个设备位置

        Args:
            mac: MAC地址

        Returns:
            是否成功
        """
        # 采集RSSI
        rssi = self.signal_collector.collect_rssi(mac)
        if rssi is None:
            return False

        # 定位
        try:
            result = self.localization_engine.locate(rssi)
            position = result['position']
            confidence = result['confidence']

            # 更新设备信息
            with self.lock:
                if mac in self.devices:
                    self.devices[mac].update(position, rssi, confidence)
                else:
                    # 自动添加新设备
                    device = TargetDevice(mac=mac)
                    device.update(position, rssi, confidence)
                    self.devices[mac] = device

            return True

        except Exception as e:
            print(f"定位设备 {mac} 失败: {e}")
            return False

    def auto_discover_and_track(self):
        """自动发现并跟踪所有设备"""
        try:
            # 扫描设备
            discovered_macs = self.signal_collector.scan_devices()

            # 添加新设备
            for mac in discovered_macs:
                if mac not in self.devices:
                    self.add_device(mac, f"Device_{mac[-5:]}")

            # 更新所有已知设备
            for mac in list(self.devices.keys()):
                self.update_device_location(mac)

        except Exception as e:
            print(f"自动发现设备失败: {e}")

    def start_tracking(self, auto_discover: bool = True):
        """
        开始跟踪

        Args:
            auto_discover: 是否自动发现新设备
        """
        if self.is_tracking:
            print("跟踪已在运行")
            return

        self.is_tracking = True

        def tracking_loop():
            print("开始实时跟踪...")
            while self.is_tracking:
                try:
                    if auto_discover:
                        self.auto_discover_and_track()
                    else:
                        # 只更新已知设备
                        for mac in list(self.devices.keys()):
                            self.update_device_location(mac)

                    time.sleep(self.update_interval)

                except Exception as e:
                    print(f"跟踪循环错误: {e}")
                    time.sleep(self.update_interval)

            print("停止实时跟踪")

        self.tracking_thread = threading.Thread(target=tracking_loop, daemon=True)
        self.tracking_thread.start()

    def stop_tracking(self):
        """停止跟踪"""
        self.is_tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=5.0)

    def get_statistics(self) -> Dict:
        """获取跟踪统计信息"""
        with self.lock:
            total_devices = len(self.devices)
            active_devices = len([d for d in self.devices.values()
                                 if d.is_active(self.device_timeout)])

            positions = []
            confidences = []
            for device in self.devices.values():
                if device.position is not None and device.is_active(self.device_timeout):
                    positions.append(device.position)
                    confidences.append(device.confidence)

            avg_confidence = np.mean(confidences) if confidences else 0.0

            return {
                'total_devices': total_devices,
                'active_devices': active_devices,
                'inactive_devices': total_devices - active_devices,
                'avg_confidence': avg_confidence,
                'tracked_positions': len(positions)
            }

    def export_trajectory(self, mac: str, format: str = 'numpy') -> Optional[np.ndarray]:
        """
        导出设备轨迹

        Args:
            mac: MAC地址
            format: 'numpy' 或 'list'

        Returns:
            轨迹数据
        """
        device = self.get_device(mac)
        if device is None:
            return None

        if format == 'numpy':
            return device.get_trajectory_array()
        elif format == 'list':
            return device.trajectory
        else:
            raise ValueError(f"不支持的格式: {format}")

    def clear_inactive_devices(self):
        """清除不活跃的设备"""
        with self.lock:
            inactive = [mac for mac, dev in self.devices.items()
                       if not dev.is_active(self.device_timeout)]
            for mac in inactive:
                del self.devices[mac]
            if inactive:
                print(f"清除 {len(inactive)} 个不活跃设备")


if __name__ == "__main__":
    print("设备跟踪模块测试")
    print("\n使用示例:")
    print("  tracker = DeviceTracker(signal_collector, localization_engine)")
    print("  tracker.add_device('AA:BB:CC:DD:EE:FF', '目标手机')")
    print("  tracker.start_tracking(auto_discover=True)")
    print("  ...")
    print("  devices = tracker.get_active_devices()")
    print("  for dev in devices:")
    print("      print(f'{dev.name}: {dev.position}')")
    print("  ...")
    print("  tracker.stop_tracking()")
