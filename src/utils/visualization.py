"""
可视化工具模块
提供3D模型、信号分布、定位结果的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Dict, Tuple
import os


class Visualizer:
    """可视化工具类"""

    def __init__(self, model=None, figsize=(12, 8)):
        """
        初始化

        Args:
            model: IndoorModel对象
            figsize: 图形尺寸
        """
        self.model = model
        self.figsize = figsize

    def plot_signal_heatmap(self, fingerprint_db, ap_index: int = 0, save_path: str = None):
        """
        绘制信号强度热图

        Args:
            fingerprint_db: FingerprintDatabase对象
            ap_index: AP索引
            save_path: 保存路径
        """
        positions, rssi_matrix = fingerprint_db.get_all_fingerprints()

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
        fig, ax = plt.subplots(figsize=self.figsize)
        contour = ax.contourf(xi, yi, zi, levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='RSSI (dBm)')

        # 标记AP位置
        if hasattr(fingerprint_db, 'ap_positions'):
            ap_pos = fingerprint_db.ap_positions[ap_index]
            ax.plot(ap_pos[0], ap_pos[1], 'r*', markersize=20, label=f'AP {ap_index}')

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'AP {ap_index} Signal Strength Heatmap', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"热图已保存到: {save_path}")

        plt.show()

    def plot_all_aps_heatmap(self, fingerprint_db, save_path: str = None):
        """
        绘制所有AP的信号强度热图

        Args:
            fingerprint_db: FingerprintDatabase对象
            save_path: 保存路径
        """
        positions, rssi_matrix = fingerprint_db.get_all_fingerprints()
        num_aps = rssi_matrix.shape[1]

        # 创建子图
        cols = 2
        rows = (num_aps + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
        axes = axes.flatten() if num_aps > 1 else [axes]

        for ap_idx in range(num_aps):
            ax = axes[ap_idx]

            # 提取数据
            x = positions[:, 0]
            y = positions[:, 1]
            rssi = rssi_matrix[:, ap_idx]

            # 创建网格和插值
            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            xi, yi = np.meshgrid(xi, yi)

            from scipy.interpolate import griddata
            zi = griddata((x, y), rssi, (xi, yi), method='cubic')

            # 绘图
            contour = ax.contourf(xi, yi, zi, levels=15, cmap='viridis')
            plt.colorbar(contour, ax=ax, label='RSSI (dBm)')

            # 标记AP位置
            if hasattr(fingerprint_db, 'ap_positions'):
                ap_pos = fingerprint_db.ap_positions[ap_idx]
                ax.plot(ap_pos[0], ap_pos[1], 'r*', markersize=15)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'AP {ap_idx}')
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(num_aps, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"所有AP热图已保存到: {save_path}")

        plt.show()

    def plot_localization_result(self, true_position: np.ndarray, estimated_position: np.ndarray,
                                  fingerprint_db=None, save_path: str = None, use_3d: bool = True,
                                  show_fingerprints: bool = True, downsample_factor: int = 10):
        """
        绘制定位结果

        Args:
            true_position: 真实位置 (3,)
            estimated_position: 估计位置 (3,)
            fingerprint_db: FingerprintDatabase对象
            save_path: 保存路径
            use_3d: 是否使用3D可视化 (默认True)
            show_fingerprints: 是否显示指纹点 (默认True，关闭可显著提升性能)
            downsample_factor: 指纹点下采样因子 (默认10，每10个点显示1个)
        """
        if use_3d:
            self._plot_localization_result_3d(true_position, estimated_position, fingerprint_db,
                                             save_path, show_fingerprints, downsample_factor)
        else:
            self._plot_localization_result_2d(true_position, estimated_position, fingerprint_db, save_path)

    def _plot_localization_result_2d(self, true_position: np.ndarray, estimated_position: np.ndarray,
                                      fingerprint_db=None, save_path: str = None):
        """绘制2D定位结果"""
        fig, ax = plt.subplots(figsize=self.figsize)

        # 绘制指纹点
        if fingerprint_db:
            positions, _ = fingerprint_db.get_all_fingerprints()
            ax.scatter(positions[:, 0], positions[:, 1], c='lightgray',
                       s=10, alpha=0.5, label='Fingerprint Points')

            # 绘制AP位置
            if hasattr(fingerprint_db, 'ap_positions'):
                for i, ap_pos in enumerate(fingerprint_db.ap_positions):
                    ax.plot(ap_pos[0], ap_pos[1], 'r^', markersize=15,
                            label=f'AP {i}' if i == 0 else '')

        # 绘制真实位置和估计位置
        ax.plot(true_position[0], true_position[1], 'go', markersize=12,
                label='True Position', markeredgecolor='black', markeredgewidth=2)
        ax.plot(estimated_position[0], estimated_position[1], 'bs', markersize=12,
                label='Estimated Position', markeredgecolor='black', markeredgewidth=2)

        # 绘制误差线
        ax.plot([true_position[0], estimated_position[0]],
                [true_position[1], estimated_position[1]],
                'k--', linewidth=2, label='Location Error')

        # 计算误差
        error = np.linalg.norm(true_position[:2] - estimated_position[:2])
        ax.text(0.02, 0.98, f'Location Error: {error:.2f} m',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Indoor Localization Result', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"定位结果图已保存到: {save_path}")

        plt.show()

    def _plot_localization_result_3d(self, true_position: np.ndarray, estimated_position: np.ndarray,
                                      fingerprint_db=None, save_path: str = None,
                                      show_fingerprints: bool = True, downsample_factor: int = 10):
        """
        绘制3D定位结果

        Args:
            true_position: 真实位置
            estimated_position: 估计位置
            fingerprint_db: 指纹库
            save_path: 保存路径
            show_fingerprints: 是否显示指纹点（关闭可提升性能）
            downsample_factor: 指纹点下采样因子（每N个点显示1个，提升性能）
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        # 绘制指纹点（优化版：下采样）
        if fingerprint_db and show_fingerprints:
            positions, _ = fingerprint_db.get_all_fingerprints()
            # 下采样以提升性能
            if len(positions) > 100:  # 仅在点数较多时下采样
                positions = positions[::downsample_factor]
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c='lightgray', s=2, alpha=0.15, label='Fingerprint Points')

        # 绘制AP位置
        if fingerprint_db and hasattr(fingerprint_db, 'ap_positions'):
            ap_positions = np.array(fingerprint_db.ap_positions)
            ax.scatter(ap_positions[:, 0], ap_positions[:, 1], ap_positions[:, 2],
                      c='red', marker='^', s=200, edgecolors='black', linewidths=2,
                      label='Access Points')

        # 绘制真实位置
        ax.scatter(true_position[0], true_position[1], true_position[2],
                  c='green', marker='o', s=200, edgecolors='black', linewidths=2,
                  label='True Position')

        # 绘制估计位置
        ax.scatter(estimated_position[0], estimated_position[1], estimated_position[2],
                  c='blue', marker='s', s=200, edgecolors='black', linewidths=2,
                  label='Estimated Position')

        # 绘制误差线（3D）
        ax.plot([true_position[0], estimated_position[0]],
               [true_position[1], estimated_position[1]],
               [true_position[2], estimated_position[2]],
               'k--', linewidth=2, label='Location Error')

        # 计算3D误差
        error_3d = np.linalg.norm(true_position - estimated_position)
        error_2d = np.linalg.norm(true_position[:2] - estimated_position[:2])

        # 添加文本信息
        info_text = f'3D Error: {error_3d:.2f} m\n2D Error: {error_2d:.2f} m\nΔZ: {abs(true_position[2] - estimated_position[2]):.2f} m'
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 设置标签和标题
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('3D Indoor Localization Result', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)

        # 设置相等的坐标轴比例（优化版：使用已有数据计算边界）
        if fingerprint_db:
            # 使用已下采样的数据或重新获取（但只计算边界，不渲染）
            all_positions, _ = fingerprint_db.get_all_fingerprints()
            x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
            y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
            z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()

            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
            mid_x = (x_max + x_min) * 0.5
            mid_y = (y_max + y_min) * 0.5
            mid_z = (z_max + z_min) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # 设置视角
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"定位结果图已保存到: {save_path}")

        plt.show()

    def plot_trajectory(self, true_trajectory: np.ndarray, estimated_trajectory: np.ndarray,
                        fingerprint_db=None, save_path: str = None, use_3d: bool = True,
                        show_fingerprints: bool = True, downsample_factor: int = 10):
        """
        绘制定位轨迹

        Args:
            true_trajectory: 真实轨迹 shape=(N, 3)
            estimated_trajectory: 估计轨迹 shape=(N, 3)
            fingerprint_db: FingerprintDatabase对象
            save_path: 保存路径
            use_3d: 是否使用3D可视化 (默认True)
            show_fingerprints: 是否显示指纹点 (默认True，关闭可显著提升性能)
            downsample_factor: 指纹点下采样因子 (默认10，每10个点显示1个)
        """
        if use_3d:
            self._plot_trajectory_3d(true_trajectory, estimated_trajectory, fingerprint_db,
                                    save_path, show_fingerprints, downsample_factor)
        else:
            self._plot_trajectory_2d(true_trajectory, estimated_trajectory, fingerprint_db, save_path)

    def _plot_trajectory_2d(self, true_trajectory: np.ndarray, estimated_trajectory: np.ndarray,
                            fingerprint_db=None, save_path: str = None):
        """绘制2D轨迹"""
        fig, ax = plt.subplots(figsize=self.figsize)

        # 绘制指纹点
        if fingerprint_db:
            positions, _ = fingerprint_db.get_all_fingerprints()
            ax.scatter(positions[:, 0], positions[:, 1], c='lightgray',
                       s=5, alpha=0.3, label='Fingerprint Points')

            # 绘制AP位置
            if hasattr(fingerprint_db, 'ap_positions'):
                for i, ap_pos in enumerate(fingerprint_db.ap_positions):
                    ax.plot(ap_pos[0], ap_pos[1], 'r^', markersize=12,
                            label='AP' if i == 0 else '')

        # 绘制轨迹
        ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'g-o',
                linewidth=2, markersize=6, label='True Trajectory', alpha=0.7)
        ax.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 'b-s',
                linewidth=2, markersize=6, label='Estimated Trajectory', alpha=0.7)

        # 标记起点和终点
        ax.plot(true_trajectory[0, 0], true_trajectory[0, 1], 'g*',
                markersize=20, label='Start Point')
        ax.plot(true_trajectory[-1, 0], true_trajectory[-1, 1], 'r*',
                markersize=20, label='End Point')

        # 计算平均误差
        errors = np.linalg.norm(true_trajectory[:, :2] - estimated_trajectory[:, :2], axis=1)
        mean_error = np.mean(errors)
        ax.text(0.02, 0.98, f'Mean Location Error: {mean_error:.2f} m',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Indoor Localization Trajectory', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"轨迹图已保存到: {save_path}")

        plt.show()

    def _plot_trajectory_3d(self, true_trajectory: np.ndarray, estimated_trajectory: np.ndarray,
                            fingerprint_db=None, save_path: str = None,
                            show_fingerprints: bool = True, downsample_factor: int = 10):
        """
        绘制3D轨迹

        Args:
            true_trajectory: 真实轨迹
            estimated_trajectory: 估计轨迹
            fingerprint_db: 指纹库
            save_path: 保存路径
            show_fingerprints: 是否显示指纹点（关闭可提升性能）
            downsample_factor: 指纹点下采样因子（每N个点显示1个，提升性能）
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        # 绘制指纹点（优化版：下采样）
        if fingerprint_db and show_fingerprints:
            positions, _ = fingerprint_db.get_all_fingerprints()
            # 下采样以提升性能
            if len(positions) > 100:
                positions = positions[::downsample_factor]
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c='lightgray', s=2, alpha=0.1, label='Fingerprint Points')

        # 绘制AP位置
        if fingerprint_db and hasattr(fingerprint_db, 'ap_positions'):
            ap_positions = np.array(fingerprint_db.ap_positions)
            ax.scatter(ap_positions[:, 0], ap_positions[:, 1], ap_positions[:, 2],
                      c='red', marker='^', s=150, edgecolors='black', linewidths=2,
                      label='Access Points')

        # 绘制轨迹
        ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2],
               'g-o', linewidth=2, markersize=6, label='True Trajectory', alpha=0.8)
        ax.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], estimated_trajectory[:, 2],
               'b-s', linewidth=2, markersize=6, label='Estimated Trajectory', alpha=0.8)

        # 标记起点和终点
        ax.scatter(true_trajectory[0, 0], true_trajectory[0, 1], true_trajectory[0, 2],
                  c='green', marker='*', s=300, edgecolors='black', linewidths=2,
                  label='Start Point')
        ax.scatter(true_trajectory[-1, 0], true_trajectory[-1, 1], true_trajectory[-1, 2],
                  c='red', marker='*', s=300, edgecolors='black', linewidths=2,
                  label='End Point')

        # 计算平均误差
        errors_3d = np.linalg.norm(true_trajectory - estimated_trajectory, axis=1)
        errors_2d = np.linalg.norm(true_trajectory[:, :2] - estimated_trajectory[:, :2], axis=1)
        mean_error_3d = np.mean(errors_3d)
        mean_error_2d = np.mean(errors_2d)

        # 添加文本信息
        info_text = f'Mean 3D Error: {mean_error_3d:.2f} m\nMean 2D Error: {mean_error_2d:.2f} m'
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 设置标签和标题
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('3D Indoor Localization Trajectory', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)

        # 设置相等的坐标轴比例（优化版）
        if fingerprint_db:
            all_positions, _ = fingerprint_db.get_all_fingerprints()
            x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
            y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
            z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()

            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
            mid_x = (x_max + x_min) * 0.5
            mid_y = (y_max + y_min) * 0.5
            mid_z = (z_max + z_min) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # 设置视角
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"轨迹图已保存到: {save_path}")

        plt.show()

    def plot_error_cdf(self, errors: np.ndarray, save_path: str = None):
        """
        绘制定位误差CDF曲线

        Args:
            errors: 误差数组
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # 排序误差
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

        # 绘制CDF
        ax.plot(sorted_errors, cdf, 'b-', linewidth=2)

        # 标记关键点
        percentiles = [50, 75, 90, 95]
        for p in percentiles:
            idx = int(len(sorted_errors) * p / 100)
            error_at_p = sorted_errors[idx]
            ax.plot([error_at_p, error_at_p], [0, p/100], 'r--', alpha=0.5)
            ax.plot([0, error_at_p], [p/100, p/100], 'r--', alpha=0.5)
            ax.text(error_at_p, p/100 + 0.02, f'{p}%: {error_at_p:.2f}m',
                    fontsize=10, ha='center')

        ax.set_xlabel('Localization Error (m)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('Localization Error Cumulative Distribution Function (CDF)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"CDF图已保存到: {save_path}")

        plt.show()


if __name__ == "__main__":
    print("可视化工具模块")
    print("\n使用示例:")
    print("  viz = Visualizer()")
    print("  viz.plot_signal_heatmap(fingerprint_db, ap_index=0)")
    print("  viz.plot_localization_result(true_pos, estimated_pos, fingerprint_db)")
