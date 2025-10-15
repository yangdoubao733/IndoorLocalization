"""
多径传播追踪模块
实现真实的多径传播模拟：追踪发射机发出的所有射线，计算所有能到达接收点的路径，并进行功率叠加
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from .ray_tracing import Ray, ReflectionPath, RayTracer


def db_to_linear(db_value: float) -> float:
    """dB值转换为线性值"""
    return 10 ** (db_value / 10.0)


def linear_to_db(linear_value: float) -> float:
    """线性值转换为dB值"""
    if linear_value <= 0:
        return -float('inf')
    return 10 * np.log10(linear_value)


class MultipathRayTracer(RayTracer):
    """
    多径射线追踪器
    扩展RayTracer，实现真实的多径传播模拟
    """

    def __init__(self, model, config):
        """
        初始化

        Args:
            model: IndoorModel对象
            config: 仿真配置
                - 基础配置同RayTracer
                - multipath_enabled: 是否启用多径追踪 (默认False)
                - num_rays: 发射射线数量 (默认360)
                - rx_tolerance: 接收点容差距离 (默认0.3米)
                - power_threshold_dbm: 功率阈值 (默认-100dBm)
        """
        super().__init__(model, config)

        self.multipath_enabled = config.get('multipath_enabled', False)
        self.num_rays = config.get('num_rays', 360)
        self.rx_tolerance = config.get('rx_tolerance', 0.3)
        self.power_threshold_dbm = config.get('power_threshold_dbm', -100.0)

        if self.multipath_enabled:
            print(f"  多径追踪: 已启用")
            print(f"    射线数量: {self.num_rays}")
            print(f"    接收容差: {self.rx_tolerance}m")
            print(f"    功率阈值: {self.power_threshold_dbm}dBm")

    def trace_all_paths_multipath(self, tx_position: np.ndarray,
                                   rx_position: np.ndarray) -> List[ReflectionPath]:
        """
        多径传播追踪：发射多个方向的射线，追踪所有能到达接收点的路径

        Args:
            tx_position: 发射机位置
            rx_position: 接收机位置

        Returns:
            所有有效路径的列表
        """
        valid_paths = []

        # 在球面上均匀生成射线方向（使用Fibonacci球面采样）
        rays = self._generate_rays_fibonacci_sphere(tx_position, self.num_rays)

        print(f"发射 {len(rays)} 条射线进行多径追踪...")

        for i, ray in enumerate(rays):
            # 追踪这条射线的所有反射路径
            paths = self._trace_single_ray_reflections(
                ray,
                rx_position,
                self.rx_tolerance,
                self.power_threshold_dbm
            )
            valid_paths.extend(paths)

            if (i + 1) % 100 == 0:
                print(f"  已处理 {i+1}/{len(rays)} 条射线, 发现 {len(valid_paths)} 条有效路径")

        print(f"多径追踪完成: 共发现 {len(valid_paths)} 条有效路径")

        return valid_paths

    def _generate_rays_fibonacci_sphere(self, origin: np.ndarray, num_rays: int) -> List[Ray]:
        """
        使用Fibonacci球面算法生成均匀分布的射线

        Args:
            origin: 射线起点
            num_rays: 射线数量

        Returns:
            射线列表
        """
        rays = []
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(num_rays):
            # Fibonacci球面采样
            theta = 2 * np.pi * i / golden_ratio
            phi = np.arccos(1 - 2 * (i + 0.5) / num_rays)

            # 转换为笛卡尔坐标
            direction = np.array([
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi)
            ])

            ray = Ray(
                origin=origin,
                direction=direction,
                power=self.config['tx_power'],
                distance=0.0,
                bounces=0
            )
            rays.append(ray)

        return rays

    def _trace_single_ray_reflections(self, ray: Ray, rx_position: np.ndarray,
                                      rx_tolerance: float,
                                      power_threshold_dbm: float) -> List[ReflectionPath]:
        """
        追踪单条射线的所有反射路径，直到能量耗尽或到达接收点

        Args:
            ray: 初始射线
            rx_position: 接收点位置
            rx_tolerance: 到达接收点的容差
            power_threshold_dbm: 功率阈值

        Returns:
            有效路径列表
        """
        valid_paths = []

        # 递归追踪（使用栈避免真正的递归）
        stack = [(ray, [ray.origin.copy()], [], 0.0, 0.0)]  # (current_ray, path_points, materials, total_distance, total_loss)

        while stack:
            current_ray, path_points, materials, total_distance, total_loss = stack.pop()

            # 计算当前功率
            current_power = self.config['tx_power'] - total_loss
            if current_power < power_threshold_dbm:
                continue  # 功率太弱，剪枝

            # 检查是否接近接收点
            distance_to_rx = np.linalg.norm(rx_position - current_ray.origin)

            # 检查方向是否大致指向接收点
            if distance_to_rx > 1e-6:
                direction_to_rx = (rx_position - current_ray.origin) / distance_to_rx
                dot_product = np.dot(current_ray.direction, direction_to_rx)

                # 如果在接收点附近且方向大致正确
                if distance_to_rx <= rx_tolerance and dot_product > 0.5:
                    # 到达接收点！
                    final_path_points = path_points + [rx_position.copy()]
                    final_distance = total_distance + distance_to_rx
                    final_fspl = self.path_loss_model.free_space_loss(distance_to_rx)
                    final_loss = total_loss + final_fspl

                    valid_paths.append(ReflectionPath(
                        total_distance=final_distance,
                        total_loss=final_loss,
                        num_bounces=len(materials),
                        path_points=final_path_points,
                        materials=materials.copy()
                    ))
                    continue

            # 如果已经达到最大反射次数
            if current_ray.bounces >= self.max_reflections:
                continue

            # 追踪射线，寻找下一个交点
            hit_point, hit_distance, is_hit, tri_index = self.trace_ray(
                current_ray,
                max_distance=100.0
            )

            if not is_hit:
                # 没有击中任何东西，检查是否能直接到达接收点
                if distance_to_rx < 100.0:
                    direction_to_rx = (rx_position - current_ray.origin) / distance_to_rx
                    dot_product = np.dot(current_ray.direction, direction_to_rx)

                    # 如果方向大致正确（夹角<30度）
                    if dot_product > 0.866:  # cos(30°)
                        final_path_points = path_points + [rx_position.copy()]
                        final_distance = total_distance + distance_to_rx
                        final_fspl = self.path_loss_model.free_space_loss(distance_to_rx)
                        final_loss = total_loss + final_fspl

                        valid_paths.append(ReflectionPath(
                            total_distance=final_distance,
                            total_loss=final_loss,
                            num_bounces=len(materials),
                            path_points=final_path_points,
                            materials=materials.copy()
                        ))
                continue

            # 击中墙壁，准备反射
            new_distance = total_distance + hit_distance
            new_path_points = path_points + [hit_point.copy()]

            # 获取材料和法向量
            material = self.get_material_at_triangle(tri_index)
            new_materials = materials + [material]
            normal = self.get_surface_normal(tri_index)

            # 计算反射系数和损耗
            reflection_coeff = self.path_loss_model.material_props.get_reflection_coefficient(material)
            incident_angle = np.arccos(np.abs(np.dot(-current_ray.direction, normal)))
            refl_loss = self.path_loss_model.reflection_loss(material, incident_angle)
            absorption_loss = self.path_loss_model.material_props.get_absorption_loss(material)

            # 总损耗：传播损耗 + 反射损耗 + 吸收损耗的一部分
            segment_fspl = self.path_loss_model.free_space_loss(hit_distance)
            new_total_loss = total_loss + segment_fspl + refl_loss + absorption_loss * 0.2  # 吸收损耗权重

            # 计算反射方向
            reflected_direction = self.calculate_reflection_direction(
                current_ray.direction, normal
            )

            # 创建反射后的新射线
            new_ray = Ray(
                origin=hit_point + reflected_direction * 1e-3,  # 微小偏移避免自交
                direction=reflected_direction,
                power=self.config['tx_power'] - new_total_loss,
                distance=new_distance,
                bounces=current_ray.bounces + 1
            )

            # 压入栈继续追踪
            stack.append((new_ray, new_path_points, new_materials, new_distance, new_total_loss))

        return valid_paths

    def combine_multipath_power(self, paths: List[ReflectionPath]) -> float:
        """
        将多条路径的功率相加（功率叠加，非dB叠加）

        Args:
            paths: 路径列表

        Returns:
            总接收功率 (dBm)
        """
        if not paths:
            return -float('inf')  # 没有路径，无信号

        total_power_linear = 0.0

        for path in paths:
            # 计算这条路径的接收功率
            path_power_dbm = self.config['tx_power'] - path.total_loss
            # 转换为线性值并累加
            path_power_linear = db_to_linear(path_power_dbm)
            total_power_linear += path_power_linear

        # 转换回dB
        total_power_dbm = linear_to_db(total_power_linear)

        return total_power_dbm

    def simulate_signal(self, tx_position: np.ndarray, rx_position: np.ndarray) -> float:
        """
        模拟两点间的信号强度（支持多径模式）

        Args:
            tx_position: 发射机位置 (3,)
            rx_position: 接收机位置 (3,)

        Returns:
            接收信号强度 (dBm)
        """
        if self.multipath_enabled:
            # 多径模式：追踪所有路径并功率叠加
            paths = self.trace_all_paths_multipath(tx_position, rx_position)
            rx_power = self.combine_multipath_power(paths)

        elif self.high_precision_mode:
            # 高精度单路径模式
            path = self.trace_ray_with_reflections(tx_position, rx_position)
            rx_power = self.config['tx_power'] - path.total_loss

        else:
            # 简化模式
            distance = np.linalg.norm(rx_position - tx_position)
            direction = (rx_position - tx_position) / distance
            hit_point, hit_distance, is_hit, _ = self.trace_ray(
                Ray(tx_position, direction, self.config['tx_power'], 0.0, 0)
            )

            if not is_hit or hit_distance >= distance:
                rx_power = self.path_loss_model.calculate_received_power(distance, 0)
            else:
                rx_power = self.path_loss_model.calculate_received_power(distance, 1)

        # 添加阴影衰落
        shadow_fading = np.random.normal(0, self.config.get('shadow_fading_std', 4.0))
        rx_power += shadow_fading

        return rx_power


def create_multipath_ray_tracer(model, config):
    """
    便捷函数: 创建多径射线追踪器

    Args:
        model: IndoorModel对象
        config: 仿真配置

    Returns:
        MultipathRayTracer对象
    """
    return MultipathRayTracer(model, config)
