"""
信号采集模块
支持从真实AP或模拟环境采集RSSI数据
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import time
import socket
import json


class SignalCollector(ABC):
    """信号采集器基类"""

    def __init__(self, ap_positions: List[Tuple[float, float, float]]):
        """
        初始化

        Args:
            ap_positions: AP位置列表 [(x,y,z), ...]
        """
        self.ap_positions = ap_positions
        self.num_aps = len(ap_positions)

    @abstractmethod
    def collect_rssi(self, target_mac: str) -> Optional[np.ndarray]:
        """
        采集目标设备的RSSI

        Args:
            target_mac: 目标设备MAC地址

        Returns:
            RSSI数组 shape=(num_aps,) 或 None（如果未检测到）
        """
        pass

    @abstractmethod
    def scan_devices(self) -> List[str]:
        """
        扫描当前环境中的所有设备

        Returns:
            设备MAC地址列表
        """
        pass


class SimulatedSignalCollector(SignalCollector):
    """模拟信号采集器（用于测试）"""

    def __init__(self, ap_positions: List[Tuple[float, float, float]],
                 fingerprint_db=None, ray_tracer=None):
        """
        初始化模拟采集器

        Args:
            ap_positions: AP位置列表
            fingerprint_db: 指纹库（可选，用于快速查询）
            ray_tracer: 射线追踪器（可选，用于实时仿真）
        """
        super().__init__(ap_positions)
        self.fingerprint_db = fingerprint_db
        self.ray_tracer = ray_tracer

        # 模拟设备列表（MAC地址 -> 位置）
        self.simulated_devices: Dict[str, np.ndarray] = {}

        print(f"模拟信号采集器初始化完成")
        print(f"  AP数量: {self.num_aps}")
        print(f"  模式: {'指纹库查询' if fingerprint_db else '实时仿真' if ray_tracer else '随机生成'}")

    def add_simulated_device(self, mac: str, position: np.ndarray):
        """
        添加模拟设备

        Args:
            mac: MAC地址
            position: 设备位置 (x,y,z)
        """
        self.simulated_devices[mac] = position
        print(f"添加模拟设备: {mac} @ {position}")

    def update_device_position(self, mac: str, position: np.ndarray):
        """更新设备位置（模拟移动）"""
        if mac in self.simulated_devices:
            self.simulated_devices[mac] = position

    def collect_rssi(self, target_mac: str) -> Optional[np.ndarray]:
        """
        采集RSSI（模拟）

        Args:
            target_mac: 目标MAC地址

        Returns:
            RSSI数组或None
        """
        if target_mac not in self.simulated_devices:
            return None

        position = self.simulated_devices[target_mac]

        # 优先使用指纹库查询（最快）
        if self.fingerprint_db is not None:
            rssi = self._get_rssi_from_fingerprint(position)
            if rssi is not None:
                # 添加测量噪声
                noise = np.random.normal(0, 2.0, size=rssi.shape)
                return rssi + noise

        # 使用射线追踪实时仿真
        if self.ray_tracer is not None:
            rssi = self._get_rssi_from_simulation(position)
            if rssi is not None:
                return rssi

        # 降级方案：基于距离的简单模型
        return self._get_rssi_from_distance_model(position)

    def _get_rssi_from_fingerprint(self, position: np.ndarray) -> Optional[np.ndarray]:
        """从指纹库查询RSSI"""
        # 查找最近的指纹点
        positions, rssi_matrix = self.fingerprint_db.get_all_fingerprints()
        distances = np.linalg.norm(positions - position, axis=1)
        nearest_idx = np.argmin(distances)

        # 如果距离太远（>2米），返回None
        if distances[nearest_idx] > 2.0:
            return None

        return rssi_matrix[nearest_idx].copy()

    def _get_rssi_from_simulation(self, position: np.ndarray) -> Optional[np.ndarray]:
        """通过射线追踪仿真RSSI"""
        rssi_values = []
        for ap_pos in self.ap_positions:
            rssi = self.ray_tracer.compute_rssi(ap_pos, position)
            rssi_values.append(rssi)
        return np.array(rssi_values)

    def _get_rssi_from_distance_model(self, position: np.ndarray) -> np.ndarray:
        """基于距离的简单路径损耗模型"""
        rssi_values = []

        # 自由空间路径损耗: RSSI = Pt - 20*log10(d) - 20*log10(f) - 32.44
        tx_power = 20.0  # dBm
        frequency = 2.4  # GHz

        for ap_pos in self.ap_positions:
            distance = np.linalg.norm(np.array(ap_pos) - position)
            distance = max(distance, 0.1)  # 避免除零

            # 路径损耗
            path_loss = 20 * np.log10(distance) + 20 * np.log10(frequency * 1000) + 32.44
            rssi = tx_power - path_loss

            # 添加阴影衰落
            rssi += np.random.normal(0, 4.0)

            rssi_values.append(rssi)

        return np.array(rssi_values)

    def scan_devices(self) -> List[str]:
        """扫描所有模拟设备"""
        return list(self.simulated_devices.keys())


class RealAPSignalCollector(SignalCollector):
    """真实AP信号采集器（连接真实硬件）"""

    def __init__(self, ap_positions: List[Tuple[float, float, float]],
                 ap_addresses: List[str],
                 ap_port: int = 9999,
                 protocol: str = 'udp'):
        """
        初始化真实AP采集器

        Args:
            ap_positions: AP位置列表
            ap_addresses: AP的IP地址列表
            ap_port: 通信端口
            protocol: 通信协议 ('udp' 或 'tcp')
        """
        super().__init__(ap_positions)
        self.ap_addresses = ap_addresses
        self.ap_port = ap_port
        self.protocol = protocol

        if len(ap_addresses) != len(ap_positions):
            raise ValueError("AP地址数量必须与位置数量一致")

        # 创建socket连接
        self.sockets = []
        for i, addr in enumerate(ap_addresses):
            try:
                if protocol == 'udp':
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                else:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((addr, ap_port))
                self.sockets.append(sock)
                print(f"已连接到 AP{i+1}: {addr}:{ap_port}")
            except Exception as e:
                print(f"警告: 无法连接到 AP{i+1} ({addr}): {e}")
                self.sockets.append(None)

        print(f"真实AP信号采集器初始化完成")
        print(f"  已连接AP数量: {sum(1 for s in self.sockets if s is not None)}/{self.num_aps}")

    def collect_rssi(self, target_mac: str) -> Optional[np.ndarray]:
        """
        从真实AP采集RSSI

        Args:
            target_mac: 目标设备MAC地址

        Returns:
            RSSI数组或None
        """
        rssi_values = []

        for i, sock in enumerate(self.sockets):
            if sock is None:
                rssi_values.append(-100.0)  # 未连接的AP使用最弱信号
                continue

            try:
                # 向AP发送查询请求
                request = {
                    'command': 'get_rssi',
                    'target_mac': target_mac
                }

                if self.protocol == 'udp':
                    sock.sendto(json.dumps(request).encode(),
                               (self.ap_addresses[i], self.ap_port))
                    sock.settimeout(1.0)
                    data, _ = sock.recvfrom(1024)
                else:
                    sock.sendall(json.dumps(request).encode() + b'\n')
                    sock.settimeout(1.0)
                    data = sock.recv(1024)

                # 解析响应
                response = json.loads(data.decode())
                rssi = response.get('rssi', -100.0)
                rssi_values.append(rssi)

            except socket.timeout:
                print(f"警告: AP{i+1} 超时")
                rssi_values.append(-100.0)
            except Exception as e:
                print(f"警告: AP{i+1} 采集失败: {e}")
                rssi_values.append(-100.0)

        rssi_array = np.array(rssi_values)

        # 如果所有AP都是最弱信号，说明设备不存在
        if np.all(rssi_array <= -99):
            return None

        return rssi_array

    def scan_devices(self) -> List[str]:
        """
        扫描环境中的所有设备

        Returns:
            设备MAC地址列表
        """
        devices_set = set()

        for i, sock in enumerate(self.sockets):
            if sock is None:
                continue

            try:
                # 向AP请求设备列表
                request = {'command': 'scan_devices'}

                if self.protocol == 'udp':
                    sock.sendto(json.dumps(request).encode(),
                               (self.ap_addresses[i], self.ap_port))
                    sock.settimeout(2.0)
                    data, _ = sock.recvfrom(4096)
                else:
                    sock.sendall(json.dumps(request).encode() + b'\n')
                    sock.settimeout(2.0)
                    data = sock.recv(4096)

                response = json.loads(data.decode())
                devices = response.get('devices', [])
                devices_set.update(devices)

            except Exception as e:
                print(f"警告: AP{i+1} 扫描失败: {e}")

        return list(devices_set)

    def __del__(self):
        """清理资源"""
        for sock in self.sockets:
            if sock is not None:
                try:
                    sock.close()
                except:
                    pass


def create_signal_collector(mode: str = 'simulated', **kwargs) -> SignalCollector:
    """
    便捷函数：创建信号采集器

    Args:
        mode: 'simulated' 或 'real'
        **kwargs: 其他参数

    Returns:
        SignalCollector对象
    """
    if mode == 'simulated':
        return SimulatedSignalCollector(**kwargs)
    elif mode == 'real':
        return RealAPSignalCollector(**kwargs)
    else:
        raise ValueError(f"不支持的模式: {mode}")


if __name__ == "__main__":
    print("信号采集模块测试")
    print("\n使用示例:")
    print("  # 模拟模式")
    print("  collector = create_signal_collector('simulated', ap_positions=[(0,0,2.5), ...])")
    print("  collector.add_simulated_device('AA:BB:CC:DD:EE:FF', np.array([5, 5, 1.5]))")
    print("  rssi = collector.collect_rssi('AA:BB:CC:DD:EE:FF')")
    print("\n  # 真实模式")
    print("  collector = create_signal_collector('real', ")
    print("      ap_positions=[(0,0,2.5), ...],")
    print("      ap_addresses=['192.168.1.101', '192.168.1.102', ...])")
    print("  rssi = collector.collect_rssi('AA:BB:CC:DD:EE:FF')")
