"""
射线追踪电磁仿真模块
基于几何光学原理模拟电磁波传播
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from scipy.constants import speed_of_light


@dataclass
class Ray:
    """射线类"""
    origin: np.ndarray      # 起点 (3,)
    direction: np.ndarray   # 方向 (3,)
    power: float           # 功率 (dBm)
    distance: float        # 传播距离 (m)
    bounces: int          # 反射次数


class PathLossModel:
    """路径损耗模型"""

    def __init__(self, frequency: float = 2.4e9, tx_power: float = 20.0):
        """
        初始化

        Args:
            frequency: 工作频率 (Hz)
            tx_power: 发射功率 (dBm)
        """
        self.frequency = frequency
        self.tx_power = tx_power
        self.wavelength = speed_of_light / frequency

    def free_space_loss(self, distance: float) -> float:
        """
        自由空间路径损耗 (Friis公式)

        Args:
            distance: 传播距离 (m)

        Returns:
            路径损耗 (dB)
        """
        if distance < 1e-6:
            return 0.0

        # FSPL = 20*log10(d) + 20*log10(f) - 147.55
        loss = 20 * np.log10(distance) + 20 * np.log10(self.frequency) - 147.55

        return loss

    def reflection_loss(self, material: str, angle: float) -> float:
        """
        反射损耗

        Args:
            material: 材料类型
            angle: 入射角 (弧度)

        Returns:
            反射损耗 (dB)
        """
        # 简化模型：不同材料的反射系数
        reflection_coefficients = {
            'concrete': 0.3,
            'brick': 0.4,
            'wood': 0.5,
            'glass': 0.7,
            'metal': 0.9,
        }

        coeff = reflection_coefficients.get(material, 0.3)

        # 考虑入射角影响 (Fresnel反射)
        angle_factor = np.cos(angle)
        effective_coeff = coeff * angle_factor

        # 转换为dB
        loss = -20 * np.log10(effective_coeff) if effective_coeff > 0 else 20.0

        return loss

    def calculate_received_power(self, distance: float, num_reflections: int = 0) -> float:
        """
        计算接收功率

        Args:
            distance: 传播距离 (m)
            num_reflections: 反射次数

        Returns:
            接收功率 (dBm)
        """
        # 自由空间损耗
        fspl = self.free_space_loss(distance)

        # 反射损耗 (简化为每次反射5dB)
        reflection_loss = num_reflections * 5.0

        # 接收功率
        rx_power = self.tx_power - fspl - reflection_loss

        return rx_power


class RayTracer:
    """射线追踪器"""

    def __init__(self, model, config: Dict):
        """
        初始化

        Args:
            model: IndoorModel对象
            config: 仿真配置
        """
        self.model = model
        self.config = config

        self.path_loss_model = PathLossModel(
            frequency=config.get('tx_frequency', 2.4e9),
            tx_power=config.get('tx_power', 20.0)
        )

        self.max_reflections = config.get('max_reflections', 3)
        self.ray_resolution = config.get('ray_resolution', 5.0)  # 角度分辨率

    def generate_rays(self, tx_position: np.ndarray, num_rays: int = 360) -> List[Ray]:
        """
        生成初始射线

        Args:
            tx_position: 发射机位置 (3,)
            num_rays: 射线数量

        Returns:
            初始射线列表
        """
        rays = []

        # 在水平面生成均匀分布的射线
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

        for angle in angles:
            direction = np.array([
                np.cos(angle),
                np.sin(angle),
                0.0
            ])

            ray = Ray(
                origin=tx_position,
                direction=direction,
                power=self.config['tx_power'],
                distance=0.0,
                bounces=0
            )

            rays.append(ray)

        return rays

    def trace_ray(self, ray: Ray, max_distance: float = 50.0) -> Tuple[np.ndarray, float, bool]:
        """
        追踪单条射线

        Args:
            ray: 射线对象
            max_distance: 最大追踪距离

        Returns:
            (hit_point, distance, is_hit): 交点位置, 距离, 是否击中
        """
        # 使用模型的射线求交功能
        ray_origins = ray.origin.reshape(1, 3)
        ray_directions = ray.direction.reshape(1, 3)

        locations, index_ray, index_tri = self.model.ray_intersect(
            ray_origins, ray_directions
        )

        if len(locations) > 0:
            # 找到最近的交点
            distances = np.linalg.norm(locations - ray.origin, axis=1)
            min_idx = np.argmin(distances)

            hit_point = locations[min_idx]
            distance = distances[min_idx]

            if distance < max_distance:
                return hit_point, distance, True

        return np.zeros(3), 0.0, False

    def simulate_signal(self, tx_position: np.ndarray, rx_position: np.ndarray) -> float:
        """
        模拟两点间的信号强度

        Args:
            tx_position: 发射机位置 (3,)
            rx_position: 接收机位置 (3,)

        Returns:
            接收信号强度 (dBm)
        """
        # 计算直线距离
        distance = np.linalg.norm(rx_position - tx_position)

        # 检查是否有直达路径 (LOS)
        direction = (rx_position - tx_position) / distance
        hit_point, hit_distance, is_hit = self.trace_ray(
            Ray(tx_position, direction, self.config['tx_power'], 0.0, 0)
        )

        if not is_hit or hit_distance >= distance:
            # 直达路径，使用自由空间损耗
            rx_power = self.path_loss_model.calculate_received_power(distance, 0)
        else:
            # 非直达路径，使用简化的多径模型
            rx_power = self.path_loss_model.calculate_received_power(distance, 1)

        # 添加阴影衰落
        shadow_fading = np.random.normal(0, self.config.get('shadow_fading_std', 4.0))
        rx_power += shadow_fading

        return rx_power

    def simulate_multi_ap(self, ap_positions: List[Tuple], rx_position: np.ndarray) -> np.ndarray:
        """
        模拟多个AP的信号强度

        Args:
            ap_positions: AP位置列表 [(x,y,z), ...]
            rx_position: 接收机位置 (3,)

        Returns:
            信号强度数组 shape=(num_aps,)
        """
        rssi_values = []

        for ap_pos in ap_positions:
            ap_pos_array = np.array(ap_pos)
            rssi = self.simulate_signal(ap_pos_array, rx_position)
            rssi_values.append(rssi)

        return np.array(rssi_values)

    def simulate_signal_batch(self, tx_positions: np.ndarray, rx_positions: np.ndarray) -> np.ndarray:
        """
        批量模拟信号强度 (向量化计算)

        Args:
            tx_positions: 发射机位置数组 shape=(M, 3) - M个AP
            rx_positions: 接收机位置数组 shape=(N, 3) - N个采样点

        Returns:
            信号强度矩阵 shape=(N, M) - 每行对应一个采样点，每列对应一个AP
        """
        num_rx = rx_positions.shape[0]
        num_tx = tx_positions.shape[0]

        # 准备批量射线数据
        # 对于每个rx点，需要向所有tx点发射射线
        ray_origins = []
        ray_directions = []
        ray_pairs = []  # 记录每条射线对应的(rx_idx, tx_idx)

        for rx_idx in range(num_rx):
            rx_pos = rx_positions[rx_idx]
            for tx_idx in range(num_tx):
                tx_pos = tx_positions[tx_idx]

                # 计算方向
                direction = tx_pos - rx_pos
                distance = np.linalg.norm(direction)
                if distance > 1e-6:
                    direction = direction / distance
                    ray_origins.append(rx_pos)
                    ray_directions.append(direction)
                    ray_pairs.append((rx_idx, tx_idx, distance))

        # 转换为numpy数组
        ray_origins = np.array(ray_origins)
        ray_directions = np.array(ray_directions)

        # 批量射线求交
        locations, index_ray, index_tri = self.model.ray_intersect(
            ray_origins, ray_directions
        )

        # 为每条射线标记是否被遮挡
        is_blocked = np.zeros(len(ray_pairs), dtype=bool)
        if len(locations) > 0:
            # 计算每个交点到射线起点的距离
            for i, ray_idx in enumerate(index_ray):
                hit_distance = np.linalg.norm(locations[i] - ray_origins[ray_idx])
                original_distance = ray_pairs[ray_idx][2]

                # 如果交点距离小于原始距离，说明被遮挡
                if hit_distance < original_distance - 1e-3:  # 1mm容差
                    is_blocked[ray_idx] = True

        # 计算所有信号强度
        rssi_matrix = np.zeros((num_rx, num_tx))

        for i, (rx_idx, tx_idx, distance) in enumerate(ray_pairs):
            # 根据是否遮挡选择反射次数
            num_reflections = 1 if is_blocked[i] else 0
            rx_power = self.path_loss_model.calculate_received_power(distance, num_reflections)

            # 添加阴影衰落
            shadow_fading = np.random.normal(0, self.config.get('shadow_fading_std', 4.0))
            rx_power += shadow_fading

            rssi_matrix[rx_idx, tx_idx] = rx_power

        return rssi_matrix


def create_ray_tracer(model, config: Dict) -> RayTracer:
    """
    便捷函数: 创建射线追踪器

    Args:
        model: IndoorModel对象
        config: 仿真配置

    Returns:
        RayTracer对象
    """
    return RayTracer(model, config)


if __name__ == "__main__":
    print("射线追踪模块测试")
    print("\n使用示例:")
    print("  from src.models import load_model")
    print("  model = load_model('data/models/indoor_scene.dae')")
    print("  tracer = create_ray_tracer(model, EM_SIMULATION_CONFIG)")
    print("  rssi = tracer.simulate_signal(tx_pos, rx_pos)")
