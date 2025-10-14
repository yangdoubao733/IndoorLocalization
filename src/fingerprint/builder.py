"""
指纹库构建模块
基于电磁仿真构建位置-信号强度映射数据库
"""

import numpy as np
import pickle
from typing import List, Dict, Tuple
from datetime import datetime
import os


class FingerprintDatabase:
    """指纹库类"""

    def __init__(self):
        self.fingerprints = {}  # {(x,y,z): [rssi1, rssi2, ...]}
        self.ap_positions = []  # AP位置列表
        self.metadata = {}      # 元数据

    def add_fingerprint(self, position: Tuple[float, float, float], rssi_values: np.ndarray):
        """
        添加指纹数据

        Args:
            position: 位置坐标 (x, y, z)
            rssi_values: RSSI值数组
        """
        # 将位置转换为可哈希的元组
        pos_key = tuple(np.round(position, 2))
        self.fingerprints[pos_key] = rssi_values

    def get_fingerprint(self, position: Tuple[float, float, float]) -> np.ndarray:
        """
        获取指定位置的指纹

        Args:
            position: 位置坐标

        Returns:
            RSSI值数组
        """
        pos_key = tuple(np.round(position, 2))
        return self.fingerprints.get(pos_key, None)

    def get_all_fingerprints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取所有指纹数据

        Returns:
            (positions, rssi_matrix): 位置数组 (N,3), RSSI矩阵 (N, num_aps)
        """
        positions = []
        rssi_values = []

        for pos, rssi in self.fingerprints.items():
            positions.append(pos)
            rssi_values.append(rssi)

        return np.array(positions), np.array(rssi_values)

    def save(self, filepath: str):
        """
        保存指纹库到文件

        Args:
            filepath: 文件路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 添加元数据
        self.metadata['created_at'] = datetime.now().isoformat()
        self.metadata['num_fingerprints'] = len(self.fingerprints)
        self.metadata['num_aps'] = len(self.ap_positions)

        data = {
            'fingerprints': self.fingerprints,
            'ap_positions': self.ap_positions,
            'metadata': self.metadata
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"指纹库已保存到: {filepath}")
        print(f"  指纹数量: {len(self.fingerprints)}")
        print(f"  AP数量: {len(self.ap_positions)}")

    @staticmethod
    def load(filepath: str) -> 'FingerprintDatabase':
        """
        从文件加载指纹库

        Args:
            filepath: 文件路径

        Returns:
            FingerprintDatabase对象
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        db = FingerprintDatabase()
        db.fingerprints = data['fingerprints']
        db.ap_positions = data['ap_positions']
        db.metadata = data.get('metadata', {})

        print(f"指纹库已加载: {filepath}")
        print(f"  指纹数量: {len(db.fingerprints)}")
        print(f"  AP数量: {len(db.ap_positions)}")

        return db


class FingerprintBuilder:
    """指纹库构建器"""

    def __init__(self, model, ray_tracer, config: Dict):
        """
        初始化

        Args:
            model: IndoorModel对象
            ray_tracer: RayTracer对象
            config: 指纹库配置
        """
        self.model = model
        self.ray_tracer = ray_tracer
        self.config = config

        self.database = FingerprintDatabase()
        self.database.ap_positions = config['ap_positions']

    def build(self, grid_spacing: float = 1.0, height: float = None,
              z_min: float = None, z_max: float = None, z_spacing: float = None,
              progress_callback=None, batch_size: int = None) -> FingerprintDatabase:
        """
        构建指纹库 (支持2D或3D网格，支持批量计算加速)

        Args:
            grid_spacing: XY平面网格间距 (米)
            height: 固定采样高度 (米)，用于2D定位。如果指定，则生成单层网格
            z_min: Z方向最小值 (米)，用于3D定位
            z_max: Z方向最大值 (米)，用于3D定位
            z_spacing: Z方向网格间距 (米)，用于3D定位
            progress_callback: 进度回调函数
            batch_size: 批量处理大小。None表示自动选择（根据点数智能分批）

        Returns:
            FingerprintDatabase对象
        """
        print("开始构建指纹库...")

        if height is not None:
            print("模式: 2D定位 (单层网格)")
        else:
            print("模式: 3D定位 (多层网格)")

        # 生成采样网格
        sampling_points = self.model.generate_sampling_grid(
            spacing=grid_spacing,
            height=height,
            z_min=z_min,
            z_max=z_max,
            z_spacing=z_spacing
        )

        total_points = len(sampling_points)
        print(f"采样点数量: {total_points}")

        # 准备AP位置数组
        ap_positions_array = np.array(self.config['ap_positions'])

        # 智能决定批量大小
        if batch_size is None:
            # 根据总点数自动选择批量大小，平衡性能和进度更新
            if total_points <= 100:
                batch_size = total_points  # 少于100个点，一次处理
            elif total_points <= 1000:
                batch_size = 50  # 100-1000个点，每批50个
            elif total_points <= 10000:
                batch_size = 100  # 1000-10000个点，每批100个
            else:
                batch_size = 200  # 超过10000个点，每批200个
            print(f"使用批量模式: 每批 {batch_size} 个点 (共 {int(np.ceil(total_points / batch_size))} 批)")
        else:
            print(f"使用批量模式: 每批 {batch_size} 个点")

        # 批量处理
        num_batches = int(np.ceil(total_points / batch_size))

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_points)
            batch_points = sampling_points[start_idx:end_idx]

            # 批量计算RSSI
            rssi_matrix = self.ray_tracer.simulate_signal_batch(
                ap_positions_array,
                batch_points
            )

            # 添加到指纹库
            for i, rx_pos in enumerate(batch_points):
                self.database.add_fingerprint(tuple(rx_pos), rssi_matrix[i])

            # 更新进度
            current = end_idx
            progress = current / total_points * 100

            # 调用回调函数（如果提供）
            if progress_callback:
                progress_callback(current, total_points, progress)

            # 命令行显示进度
            bar_length = 40
            filled_length = int(bar_length * current / total_points)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\r  进度: |{bar}| {progress:.1f}% ({current}/{total_points}) [批次 {batch_idx+1}/{num_batches}]", end='', flush=True)

        print()  # 换行

        print("指纹库构建完成!")

        return self.database

    def visualize_fingerprint(self, ap_index: int = 0):
        """
        可视化指定AP的信号强度分布

        Args:
            ap_index: AP索引
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm

            positions, rssi_matrix = self.database.get_all_fingerprints()

            # 提取X, Y坐标和RSSI值
            x = positions[:, 0]
            y = positions[:, 1]
            rssi = rssi_matrix[:, ap_index]

            # 创建网格
            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            xi, yi = np.meshgrid(xi, yi)

            # 插值
            from scipy.interpolate import griddata
            zi = griddata((x, y), rssi, (xi, yi), method='cubic')

            # 绘图
            fig, ax = plt.subplots(figsize=(10, 8))
            contour = ax.contourf(xi, yi, zi, levels=15, cmap='viridis')
            plt.colorbar(contour, ax=ax, label='RSSI (dBm)')

            # 标记AP位置
            ap_pos = self.config['ap_positions'][ap_index]
            ax.plot(ap_pos[0], ap_pos[1], 'r*', markersize=15, label=f'AP {ap_index}')

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'AP {ap_index} Signal Strength Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("需要安装 matplotlib 才能可视化")


def build_fingerprint_database(model, ray_tracer, config: Dict, batch_size: int = None) -> FingerprintDatabase:
    """
    便捷函数: 构建指纹库

    Args:
        model: IndoorModel对象
        ray_tracer: RayTracer对象
        config: 指纹库配置
            - grid_spacing: XY平面网格间距 (米)
            - height: 固定采样高度 (米)，用于2D定位
            - z_min: Z方向最小值 (米)，用于3D定位
            - z_max: Z方向最大值 (米)，用于3D定位
            - z_spacing: Z方向网格间距 (米)，用于3D定位
        batch_size: 批量处理大小，None表示一次处理所有点

    Returns:
        FingerprintDatabase对象
    """
    builder = FingerprintBuilder(model, ray_tracer, config)
    return builder.build(
        grid_spacing=config.get('grid_spacing', 1.0),
        height=config.get('height', None),  # None表示使用3D模式
        z_min=config.get('z_min', None),
        z_max=config.get('z_max', None),
        z_spacing=config.get('z_spacing', None),
        batch_size=batch_size
    )


if __name__ == "__main__":
    print("指纹库构建模块测试")
    print("\n使用示例:")
    print("  from src.models import load_model")
    print("  from src.simulation import create_ray_tracer")
    print("  model = load_model('data/models/indoor_scene.dae')")
    print("  tracer = create_ray_tracer(model, EM_SIMULATION_CONFIG)")
    print("  db = build_fingerprint_database(model, tracer, FINGERPRINT_CONFIG)")
    print("  db.save('data/fingerprints/fingerprint.pkl')")
