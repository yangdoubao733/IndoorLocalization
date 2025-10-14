"""
通用电磁信号采集模块
支持WiFi、蓝牙、手机信号、RFID等多种电磁信号源
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import time


class SignalType:
    """信号类型定义"""
    WIFI = 'WiFi'              # WiFi (2.4GHz/5GHz)
    BLUETOOTH = 'Bluetooth'    # 蓝牙 (2.4GHz)
    CELLULAR = 'Cellular'      # 手机信号 (700MHz-2600MHz)
    RFID = 'RFID'              # RFID (13.56MHz/915MHz)
    ZIGBEE = 'ZigBee'          # ZigBee (2.4GHz)
    LORA = 'LoRa'              # LoRa (433MHz/868MHz/915MHz)
    UWB = 'UWB'                # 超宽带 (3.1-10.6GHz)
    CUSTOM = 'Custom'          # 自定义频率

    # 信号参数配置
    SIGNAL_PARAMS = {
        WIFI: {'频率': 2.4e9, '发射功率': 20.0, '路径损耗指数': 2.0},
        BLUETOOTH: {'频率': 2.4e9, '发射功率': 4.0, '路径损耗指数': 2.0},
        CELLULAR: {'频率': 1.8e9, '发射功率': 23.0, '路径损耗指数': 2.5},
        RFID: {'频率': 915e6, '发射功率': 30.0, '路径损耗指数': 2.2},
        ZIGBEE: {'频率': 2.4e9, '发射功率': 0.0, '路径损耗指数': 2.0},
        LORA: {'频率': 915e6, '发射功率': 14.0, '路径损耗指数': 2.5},
        UWB: {'频率': 6.5e9, '发射功率': -10.0, '路径损耗指数': 1.8},
    }


class EMTarget:
    """电磁信号源目标"""

    def __init__(self, mac: str = None, identifier: str = None,
                 signal_type: str = SignalType.WIFI,
                 position: Optional[np.ndarray] = None,
                 frequency: Optional[float] = None,
                 tx_power: Optional[float] = None,
                 path_loss_exponent: Optional[float] = None):
        """
        初始化

        Args:
            mac: 设备MAC地址（向后兼容，等同于identifier）
            identifier: 设备标识（MAC地址、IMEI、RFID标签ID等）
            signal_type: 信号类型
            position: 设备位置 (x,y,z)
            frequency: 自定义频率（Hz）
            tx_power: 自定义发射功率（dBm）
            path_loss_exponent: 自定义路径损耗指数
        """
        # 兼容mac和identifier两种参数名
        if mac is not None:
            self.identifier = mac
            self.mac = mac
        elif identifier is not None:
            self.identifier = identifier
            self.mac = identifier
        else:
            raise ValueError("必须提供mac或identifier参数")

        self.signal_type = signal_type
        self.position = position

        # 获取信号参数
        if signal_type in SignalType.SIGNAL_PARAMS:
            params = SignalType.SIGNAL_PARAMS[signal_type]
            self.frequency = frequency if frequency is not None else params['频率']
            self.tx_power = tx_power if tx_power is not None else params['发射功率']
            self.path_loss_exponent = path_loss_exponent if path_loss_exponent is not None else params['路径损耗指数']
        else:
            # 自定义信号
            self.frequency = frequency if frequency is not None else 2.4e9
            self.tx_power = tx_power if tx_power is not None else 20.0
            self.path_loss_exponent = path_loss_exponent if path_loss_exponent is not None else 2.0

    def __repr__(self):
        return f"EMTarget({self.identifier}, {self.signal_type}, freq={self.frequency/1e9:.2f}GHz)"


class UniversalEMSignalCollector:
    """通用电磁信号采集器（模拟模式）"""

    def __init__(self, receiver_positions: List[Tuple[float, float, float]],
                 fingerprint_db=None, ray_tracer=None):
        """
        初始化

        Args:
            receiver_positions: 接收器位置列表 [(x,y,z), ...]
            fingerprint_db: 指纹库（可选）
            ray_tracer: 射线追踪器（可选）
        """
        self.receiver_positions = receiver_positions
        self.num_receivers = len(receiver_positions)
        self.fingerprint_db = fingerprint_db
        self.ray_tracer = ray_tracer

        # 电磁信号源列表
        self.em_targets: Dict[str, EMTarget] = {}

        print(f"通用电磁信号采集器初始化完成")
        print(f"  接收器数量: {self.num_receivers}")
        print(f"  支持信号类型: {', '.join([SignalType.WIFI, SignalType.BLUETOOTH, SignalType.CELLULAR, SignalType.RFID, SignalType.ZIGBEE, SignalType.LORA, SignalType.UWB])}")

    def add_em_target(self, target: EMTarget):
        """
        添加电磁信号源

        Args:
            target: EMTarget对象
        """
        self.em_targets[target.identifier] = target
        print(f"添加电磁信号源: {target}")

    def add_target(self, target: EMTarget):
        """
        添加电磁信号源（add_em_target的别名，向后兼容）

        Args:
            target: EMTarget对象
        """
        self.add_em_target(target)

    def add_target_simple(self, identifier: str, signal_type: str, position: np.ndarray):
        """
        简化添加方式

        Args:
            identifier: 设备标识
            signal_type: 信号类型
            position: 位置 (x,y,z)
        """
        target = EMTarget(identifier, signal_type, position)
        self.add_em_target(target)

    def update_target_position(self, identifier: str, position: np.ndarray):
        """更新目标位置（模拟移动）"""
        if identifier in self.em_targets:
            self.em_targets[identifier].position = position

    def collect_rssi(self, identifier: str) -> Optional[np.ndarray]:
        """
        采集信号强度

        Args:
            identifier: 目标标识

        Returns:
            RSSI数组 shape=(num_receivers,) 或 None
        """
        if identifier not in self.em_targets:
            return None

        target = self.em_targets[identifier]

        if target.position is None:
            return None

        # 方法1: 从指纹库查询（如果可用）
        if self.fingerprint_db is not None:
            rssi = self._get_rssi_from_fingerprint(target.position)
            if rssi is not None:
                # 添加测量噪声
                noise = np.random.normal(0, 2.0, size=rssi.shape)
                return rssi + noise

        # 方法2: 使用射线追踪仿真（如果可用）
        if self.ray_tracer is not None:
            rssi = self._get_rssi_from_simulation(target)
            if rssi is not None:
                return rssi

        # 方法3: 使用路径损耗模型（默认）
        return self._get_rssi_from_path_loss_model(target)

    def _get_rssi_from_fingerprint(self, position: np.ndarray) -> Optional[np.ndarray]:
        """从指纹库查询RSSI"""
        positions, rssi_matrix = self.fingerprint_db.get_all_fingerprints()
        distances = np.linalg.norm(positions - position, axis=1)
        nearest_idx = np.argmin(distances)

        # 如果距离太远（>2米），返回None
        if distances[nearest_idx] > 2.0:
            return None

        return rssi_matrix[nearest_idx].copy()

    def _get_rssi_from_simulation(self, target: EMTarget) -> Optional[np.ndarray]:
        """通过射线追踪仿真RSSI"""
        rssi_values = []
        for rx_pos in self.receiver_positions:
            rssi = self.ray_tracer.compute_rssi(rx_pos, target.position)
            rssi_values.append(rssi)
        return np.array(rssi_values)

    def _get_rssi_from_path_loss_model(self, target: EMTarget) -> np.ndarray:
        """
        基于路径损耗模型计算RSSI

        Friis公式: RSSI(dBm) = Pt - PL
        其中 PL = 20*log10(d) + 20*log10(f) + 20*log10(4π/c) + X_σ
        """
        rssi_values = []

        for rx_pos in self.receiver_positions:
            distance = np.linalg.norm(np.array(rx_pos) - target.position)
            distance = max(distance, 0.1)  # 避免除零

            # 计算路径损耗
            # 使用简化公式: PL(dB) = PL0 + 10*n*log10(d/d0) + X_σ
            d0 = 1.0  # 参考距离 1米
            frequency_ghz = target.frequency / 1e9

            # 参考距离的路径损耗
            pl0 = 20 * np.log10(4 * np.pi * d0 * frequency_ghz * 1e9 / 3e8)

            # 总路径损耗
            path_loss = pl0 + 10 * target.path_loss_exponent * np.log10(distance / d0)

            # 计算RSSI
            rssi = target.tx_power - path_loss

            # 添加阴影衰落（对数正态分布）
            shadow_fading_std = 4.0 + 2.0 * (target.path_loss_exponent - 2.0)  # 环境越复杂，标准差越大
            rssi += np.random.normal(0, shadow_fading_std)

            rssi_values.append(rssi)

        return np.array(rssi_values)

    def scan_targets(self, signal_type: Optional[str] = None) -> List[str]:
        """
        扫描所有电磁信号源

        Args:
            signal_type: 可选，只扫描特定类型的信号

        Returns:
            设备标识列表
        """
        if signal_type is None:
            return list(self.em_targets.keys())
        else:
            return [id for id, target in self.em_targets.items()
                   if target.signal_type == signal_type]

    def scan_devices(self) -> List[str]:
        """
        扫描所有设备（scan_targets的别名，向后兼容）

        Returns:
            设备标识列表
        """
        return self.scan_targets()

    def get_target_info(self, identifier: str) -> Optional[Dict]:
        """获取目标详细信息"""
        if identifier not in self.em_targets:
            return None

        target = self.em_targets[identifier]
        return {
            '标识': target.identifier,
            '信号类型': target.signal_type,
            '频率': f"{target.frequency/1e9:.3f} GHz",
            '发射功率': f"{target.tx_power:.1f} dBm",
            '位置': target.position.tolist() if target.position is not None else None
        }

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            '总目标数': len(self.em_targets),
            '按类型统计': {}
        }

        for target in self.em_targets.values():
            signal_type = target.signal_type
            if signal_type not in stats['按类型统计']:
                stats['按类型统计'][signal_type] = 0
            stats['按类型统计'][signal_type] += 1

        return stats


class RealEMSignalCollector:
    """真实电磁信号采集器（连接真实硬件）"""

    def __init__(self, receiver_positions: List[Tuple[float, float, float]],
                 receiver_addresses: List[str],
                 receiver_port: int = 9999,
                 protocol: str = 'udp'):
        """
        初始化

        Args:
            receiver_positions: 接收器位置列表
            receiver_addresses: 接收器IP地址列表
            receiver_port: 通信端口
            protocol: 通信协议 ('udp' 或 'tcp')
        """
        self.receiver_positions = receiver_positions
        self.receiver_addresses = receiver_addresses
        self.receiver_port = receiver_port
        self.protocol = protocol
        self.num_receivers = len(receiver_positions)

        if len(receiver_addresses) != len(receiver_positions):
            raise ValueError("接收器地址数量必须与位置数量一致")

        # 创建socket连接
        import socket
        import json

        self.sockets = []
        for i, addr in enumerate(receiver_addresses):
            try:
                if protocol == 'udp':
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                else:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((addr, receiver_port))
                self.sockets.append(sock)
                print(f"已连接到接收器{i+1}: {addr}:{receiver_port}")
            except Exception as e:
                print(f"警告: 无法连接到接收器{i+1} ({addr}): {e}")
                self.sockets.append(None)

        print(f"真实电磁信号采集器初始化完成")
        print(f"  已连接接收器数量: {sum(1 for s in self.sockets if s is not None)}/{self.num_receivers}")

    def collect_rssi(self, identifier: str, signal_type: str = SignalType.WIFI) -> Optional[np.ndarray]:
        """
        从真实接收器采集RSSI

        Args:
            identifier: 目标标识
            signal_type: 信号类型

        Returns:
            RSSI数组或None
        """
        import socket
        import json

        rssi_values = []

        for i, sock in enumerate(self.sockets):
            if sock is None:
                rssi_values.append(-100.0)
                continue

            try:
                # 向接收器发送查询请求
                request = {
                    'command': 'get_rssi',
                    'target_id': identifier,
                    'signal_type': signal_type
                }

                if self.protocol == 'udp':
                    sock.sendto(json.dumps(request).encode(),
                               (self.receiver_addresses[i], self.receiver_port))
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
                print(f"警告: 接收器{i+1} 超时")
                rssi_values.append(-100.0)
            except Exception as e:
                print(f"警告: 接收器{i+1} 采集失败: {e}")
                rssi_values.append(-100.0)

        rssi_array = np.array(rssi_values)

        # 如果所有接收器都是最弱信号，说明设备不存在
        if np.all(rssi_array <= -99):
            return None

        return rssi_array

    def scan_targets(self, signal_type: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        扫描环境中的所有电磁信号源

        Args:
            signal_type: 可选，只扫描特定类型的信号

        Returns:
            (标识, 信号类型) 元组列表
        """
        import socket
        import json

        targets_set = set()

        for i, sock in enumerate(self.sockets):
            if sock is None:
                continue

            try:
                # 向接收器请求设备列表
                request = {
                    'command': 'scan_targets',
                    'signal_type': signal_type
                }

                if self.protocol == 'udp':
                    sock.sendto(json.dumps(request).encode(),
                               (self.receiver_addresses[i], self.receiver_port))
                    sock.settimeout(2.0)
                    data, _ = sock.recvfrom(4096)
                else:
                    sock.sendall(json.dumps(request).encode() + b'\n')
                    sock.settimeout(2.0)
                    data = sock.recv(4096)

                response = json.loads(data.decode())
                targets = response.get('targets', [])  # [(id, signal_type), ...]
                targets_set.update(targets)

            except Exception as e:
                print(f"警告: 接收器{i+1} 扫描失败: {e}")

        return list(targets_set)

    def scan_devices(self) -> List[str]:
        """
        扫描所有设备（scan_targets的简化版本，向后兼容）

        Returns:
            设备标识列表（只返回ID，不返回信号类型）
        """
        targets = self.scan_targets()
        # 从 [(id, signal_type), ...] 提取出 [id, ...]
        if targets and isinstance(targets[0], tuple):
            return [t[0] for t in targets]
        return targets

    def __del__(self):
        """清理资源"""
        for sock in self.sockets:
            if sock is not None:
                try:
                    sock.close()
                except:
                    pass


if __name__ == "__main__":
    print("通用电磁信号采集模块测试")
    print("\n支持的信号类型:")
    for signal_type, params in SignalType.SIGNAL_PARAMS.items():
        print(f"  {signal_type}: {params['频率']/1e9:.2f} GHz, {params['发射功率']} dBm")

    print("\n使用示例:")
    print("  # 创建采集器")
    print("  collector = UniversalEMSignalCollector(receiver_positions=[(0,0,2.5), ...])")
    print("  ")
    print("  # 添加WiFi设备")
    print("  collector.add_target_simple('AA:BB:CC:DD:EE:FF', SignalType.WIFI, np.array([5,5,1.5]))")
    print("  ")
    print("  # 添加蓝牙设备")
    print("  collector.add_target_simple('12:34:56:78:90:AB', SignalType.BLUETOOTH, np.array([10,10,1.5]))")
    print("  ")
    print("  # 添加手机")
    print("  collector.add_target_simple('IMEI:123456789', SignalType.CELLULAR, np.array([3,7,1.5]))")
    print("  ")
    print("  # 采集信号")
    print("  rssi = collector.collect_rssi('AA:BB:CC:DD:EE:FF')")
