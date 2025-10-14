"""
高性能可视化工具模块 (基于Plotly)
提供GPU加速的交互式3D可视化功能
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple
import os


class VisualizerPlotly:
    """高性能可视化工具类 (使用Plotly)"""

    def __init__(self, model=None):
        """
        初始化

        Args:
            model: IndoorModel对象
        """
        self.model = model

    def _add_model_to_figure(self, fig):
        """
        将3D模型添加到图形中

        Args:
            fig: plotly Figure对象
        """
        if self.model is None or self.model.mesh is None:
            return

        try:
            # 提取模型的所有三角面片
            # IndoorModel.mesh 是 trimesh.Trimesh 对象
            vertices = self.model.mesh.vertices
            faces = self.model.mesh.faces

            # 创建3D网格
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightblue',
                opacity=0.3,
                name='室内模型',
                hoverinfo='skip',
                showlegend=True
            ))

            print("已添加3D模型到可视化")
        except Exception as e:
            print(f"添加模型失败: {e}")

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

        # 创建2D热图
        fig = go.Figure(data=go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=8,
                color=rssi,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="RSSI (dBm)"),
                line=dict(width=0)
            ),
            text=[f'RSSI: {r:.1f} dBm' for r in rssi],
            hoverinfo='text+x+y'
        ))

        # 标记AP位置
        if hasattr(fingerprint_db, 'ap_positions'):
            ap_pos = fingerprint_db.ap_positions[ap_index]
            fig.add_trace(go.Scatter(
                x=[ap_pos[0]],
                y=[ap_pos[1]],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='black')),
                name=f'AP {ap_index}',
                hoverinfo='name'
            ))

        fig.update_layout(
            title=f'AP {ap_index} Signal Strength Heatmap',
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            showlegend=True,
            hovermode='closest',
            width=1000,
            height=800
        )

        fig.update_xaxes(scaleanchor="y", scaleratio=1)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"热图已保存到: {save_path}")

        fig.show()

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

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f'AP {i}' for i in range(num_aps)],
            horizontal_spacing=0.1,
            vertical_spacing=0.1
        )

        for ap_idx in range(num_aps):
            row = ap_idx // cols + 1
            col = ap_idx % cols + 1

            x = positions[:, 0]
            y = positions[:, 1]
            rssi = rssi_matrix[:, ap_idx]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=rssi,
                        colorscale='Viridis',
                        showscale=False,
                        line=dict(width=0)
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

            # 标记AP位置
            if hasattr(fingerprint_db, 'ap_positions'):
                ap_pos = fingerprint_db.ap_positions[ap_idx]
                fig.add_trace(
                    go.Scatter(
                        x=[ap_pos[0]],
                        y=[ap_pos[1]],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='star'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )

        fig.update_layout(
            title_text="All APs Signal Strength Heatmap",
            height=400 * rows,
            width=1200
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"所有AP热图已保存到: {save_path}")

        fig.show()

    def plot_localization_result(self, true_position: np.ndarray, estimated_position: np.ndarray,
                                  fingerprint_db=None, save_path: str = None, use_3d: bool = True,
                                  show_fingerprints: bool = True, downsample_factor: int = 10):
        """
        绘制定位结果 (高性能版本)

        Args:
            true_position: 真实位置 (3,)
            estimated_position: 估计位置 (3,)
            fingerprint_db: FingerprintDatabase对象
            save_path: 保存路径
            use_3d: 是否使用3D可视化 (默认True)
            show_fingerprints: 是否显示指纹点 (默认True)
            downsample_factor: 指纹点下采样因子 (默认10)
        """
        if use_3d:
            self._plot_localization_result_3d(true_position, estimated_position, fingerprint_db,
                                             save_path, show_fingerprints, downsample_factor)
        else:
            self._plot_localization_result_2d(true_position, estimated_position, fingerprint_db, save_path)

    def _plot_localization_result_2d(self, true_position: np.ndarray, estimated_position: np.ndarray,
                                      fingerprint_db=None, save_path: str = None):
        """绘制2D定位结果"""
        fig = go.Figure()

        # 绘制指纹点
        if fingerprint_db:
            positions, _ = fingerprint_db.get_all_fingerprints()
            fig.add_trace(go.Scatter(
                x=positions[:, 0],
                y=positions[:, 1],
                mode='markers',
                marker=dict(size=3, color='lightgray', opacity=0.5),
                name='Fingerprint Points',
                hoverinfo='skip'
            ))

            # 绘制AP位置
            if hasattr(fingerprint_db, 'ap_positions'):
                ap_positions = np.array(fingerprint_db.ap_positions)
                fig.add_trace(go.Scatter(
                    x=ap_positions[:, 0],
                    y=ap_positions[:, 1],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='triangle-up', line=dict(width=2, color='black')),
                    name='Access Points',
                    hoverinfo='name'
                ))

        # 绘制真实位置
        fig.add_trace(go.Scatter(
            x=[true_position[0]],
            y=[true_position[1]],
            mode='markers',
            marker=dict(size=15, color='green', symbol='circle', line=dict(width=2, color='black')),
            name='True Position',
            hoverinfo='name'
        ))

        # 绘制估计位置
        fig.add_trace(go.Scatter(
            x=[estimated_position[0]],
            y=[estimated_position[1]],
            mode='markers',
            marker=dict(size=15, color='blue', symbol='square', line=dict(width=2, color='black')),
            name='Estimated Position',
            hoverinfo='name'
        ))

        # 绘制误差线
        fig.add_trace(go.Scatter(
            x=[true_position[0], estimated_position[0]],
            y=[true_position[1], estimated_position[1]],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='Location Error',
            hoverinfo='skip'
        ))

        # 计算误差
        error = np.linalg.norm(true_position[:2] - estimated_position[:2])

        fig.update_layout(
            title=f'Indoor Localization Result<br>Location Error: {error:.2f} m',
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            showlegend=True,
            hovermode='closest',
            width=1000,
            height=800
        )

        fig.update_xaxes(scaleanchor="y", scaleratio=1)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"定位结果图已保存到: {save_path}")

        fig.show()

    def _plot_localization_result_3d(self, true_position: np.ndarray, estimated_position: np.ndarray,
                                      fingerprint_db=None, save_path: str = None,
                                      show_fingerprints: bool = True, downsample_factor: int = 10):
        """
        绘制3D定位结果 (高性能WebGL版本)

        使用Plotly的WebGL渲染，可以流畅处理大量数据点
        """
        fig = go.Figure()

        # 添加3D模型
        self._add_model_to_figure(fig)

        # 绘制指纹点 (使用WebGL加速)
        if fingerprint_db and show_fingerprints:
            positions, _ = fingerprint_db.get_all_fingerprints()
            # 下采样
            if len(positions) > 100:
                positions = positions[::downsample_factor]

            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='lightgray',
                    opacity=0.2,
                    line=dict(width=0)
                ),
                name='Fingerprint Points',
                hoverinfo='skip',
                showlegend=True
            ))

        # 绘制AP位置
        if fingerprint_db and hasattr(fingerprint_db, 'ap_positions'):
            ap_positions = np.array(fingerprint_db.ap_positions)
            fig.add_trace(go.Scatter3d(
                x=ap_positions[:, 0],
                y=ap_positions[:, 1],
                z=ap_positions[:, 2],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                name='Access Points',
                text=[f'AP {i}' for i in range(len(ap_positions))],
                hoverinfo='text'
            ))

        # 绘制真实位置
        fig.add_trace(go.Scatter3d(
            x=[true_position[0]],
            y=[true_position[1]],
            z=[true_position[2]],
            mode='markers',
            marker=dict(
                size=10,
                color='green',
                symbol='circle',
                line=dict(width=2, color='black')
            ),
            name='True Position',
            text=f'True: ({true_position[0]:.2f}, {true_position[1]:.2f}, {true_position[2]:.2f})',
            hoverinfo='text'
        ))

        # 绘制估计位置
        fig.add_trace(go.Scatter3d(
            x=[estimated_position[0]],
            y=[estimated_position[1]],
            z=[estimated_position[2]],
            mode='markers',
            marker=dict(
                size=10,
                color='blue',
                symbol='square',
                line=dict(width=2, color='black')
            ),
            name='Estimated Position',
            text=f'Estimated: ({estimated_position[0]:.2f}, {estimated_position[1]:.2f}, {estimated_position[2]:.2f})',
            hoverinfo='text'
        ))

        # 绘制误差线
        fig.add_trace(go.Scatter3d(
            x=[true_position[0], estimated_position[0]],
            y=[true_position[1], estimated_position[1]],
            z=[true_position[2], estimated_position[2]],
            mode='lines',
            line=dict(color='black', width=4, dash='dash'),
            name='Location Error',
            hoverinfo='skip'
        ))

        # 计算误差
        error_3d = np.linalg.norm(true_position - estimated_position)
        error_2d = np.linalg.norm(true_position[:2] - estimated_position[:2])

        # 设置布局
        fig.update_layout(
            title=f'3D Indoor Localization Result<br>3D Error: {error_3d:.2f} m | 2D Error: {error_2d:.2f} m | ΔZ: {abs(true_position[2] - estimated_position[2]):.2f} m',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            showlegend=True,
            hovermode='closest',
            width=1200,
            height=900
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"定位结果图已保存到: {save_path}")

        fig.show()

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
            show_fingerprints: 是否显示指纹点
            downsample_factor: 指纹点下采样因子
        """
        if use_3d:
            self._plot_trajectory_3d(true_trajectory, estimated_trajectory, fingerprint_db,
                                    save_path, show_fingerprints, downsample_factor)
        else:
            self._plot_trajectory_2d(true_trajectory, estimated_trajectory, fingerprint_db, save_path)

    def _plot_trajectory_2d(self, true_trajectory: np.ndarray, estimated_trajectory: np.ndarray,
                            fingerprint_db=None, save_path: str = None):
        """绘制2D轨迹"""
        fig = go.Figure()

        # 绘制指纹点
        if fingerprint_db:
            positions, _ = fingerprint_db.get_all_fingerprints()
            fig.add_trace(go.Scatter(
                x=positions[:, 0],
                y=positions[:, 1],
                mode='markers',
                marker=dict(size=2, color='lightgray', opacity=0.3),
                name='Fingerprint Points',
                hoverinfo='skip'
            ))

            # 绘制AP位置
            if hasattr(fingerprint_db, 'ap_positions'):
                ap_positions = np.array(fingerprint_db.ap_positions)
                fig.add_trace(go.Scatter(
                    x=ap_positions[:, 0],
                    y=ap_positions[:, 1],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='triangle-up'),
                    name='Access Points',
                    hoverinfo='name'
                ))

        # 绘制真实轨迹
        fig.add_trace(go.Scatter(
            x=true_trajectory[:, 0],
            y=true_trajectory[:, 1],
            mode='lines+markers',
            line=dict(color='green', width=3),
            marker=dict(size=6, color='green'),
            name='True Trajectory',
            hoverinfo='x+y'
        ))

        # 绘制估计轨迹
        fig.add_trace(go.Scatter(
            x=estimated_trajectory[:, 0],
            y=estimated_trajectory[:, 1],
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=6, color='blue', symbol='square'),
            name='Estimated Trajectory',
            hoverinfo='x+y'
        ))

        # 标记起点和终点
        fig.add_trace(go.Scatter(
            x=[true_trajectory[0, 0]],
            y=[true_trajectory[0, 1]],
            mode='markers',
            marker=dict(size=20, color='green', symbol='star'),
            name='Start Point',
            hoverinfo='name'
        ))

        fig.add_trace(go.Scatter(
            x=[true_trajectory[-1, 0]],
            y=[true_trajectory[-1, 1]],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star'),
            name='End Point',
            hoverinfo='name'
        ))

        # 计算平均误差
        errors = np.linalg.norm(true_trajectory[:, :2] - estimated_trajectory[:, :2], axis=1)
        mean_error = np.mean(errors)

        fig.update_layout(
            title=f'Indoor Localization Trajectory<br>Mean Location Error: {mean_error:.2f} m',
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            showlegend=True,
            hovermode='closest',
            width=1000,
            height=800
        )

        fig.update_xaxes(scaleanchor="y", scaleratio=1)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"轨迹图已保存到: {save_path}")

        fig.show()

    def _plot_trajectory_3d(self, true_trajectory: np.ndarray, estimated_trajectory: np.ndarray,
                            fingerprint_db=None, save_path: str = None,
                            show_fingerprints: bool = True, downsample_factor: int = 10):
        """绘制3D轨迹 (高性能版本)"""
        fig = go.Figure()

        # 添加3D模型
        self._add_model_to_figure(fig)

        # 绘制指纹点
        if fingerprint_db and show_fingerprints:
            positions, _ = fingerprint_db.get_all_fingerprints()
            if len(positions) > 100:
                positions = positions[::downsample_factor]

            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(size=1.5, color='lightgray', opacity=0.15, line=dict(width=0)),
                name='Fingerprint Points',
                hoverinfo='skip'
            ))

        # 绘制AP位置
        if fingerprint_db and hasattr(fingerprint_db, 'ap_positions'):
            ap_positions = np.array(fingerprint_db.ap_positions)
            fig.add_trace(go.Scatter3d(
                x=ap_positions[:, 0],
                y=ap_positions[:, 1],
                z=ap_positions[:, 2],
                mode='markers',
                marker=dict(size=8, color='red', symbol='diamond', line=dict(width=2, color='black')),
                name='Access Points',
                text=[f'AP {i}' for i in range(len(ap_positions))],
                hoverinfo='text'
            ))

        # 绘制真实轨迹
        fig.add_trace(go.Scatter3d(
            x=true_trajectory[:, 0],
            y=true_trajectory[:, 1],
            z=true_trajectory[:, 2],
            mode='lines+markers',
            line=dict(color='green', width=4),
            marker=dict(size=4, color='green'),
            name='True Trajectory',
            hoverinfo='x+y+z'
        ))

        # 绘制估计轨迹
        fig.add_trace(go.Scatter3d(
            x=estimated_trajectory[:, 0],
            y=estimated_trajectory[:, 1],
            z=estimated_trajectory[:, 2],
            mode='lines+markers',
            line=dict(color='blue', width=4),
            marker=dict(size=4, color='blue', symbol='square'),
            name='Estimated Trajectory',
            hoverinfo='x+y+z'
        ))

        # 标记起点和终点
        fig.add_trace(go.Scatter3d(
            x=[true_trajectory[0, 0]],
            y=[true_trajectory[0, 1]],
            z=[true_trajectory[0, 2]],
            mode='markers',
            marker=dict(size=12, color='green', symbol='diamond', line=dict(width=2, color='black')),
            name='Start Point',
            text='Start',
            hoverinfo='text'
        ))

        fig.add_trace(go.Scatter3d(
            x=[true_trajectory[-1, 0]],
            y=[true_trajectory[-1, 1]],
            z=[true_trajectory[-1, 2]],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond', line=dict(width=2, color='black')),
            name='End Point',
            text='End',
            hoverinfo='text'
        ))

        # 计算误差
        errors_3d = np.linalg.norm(true_trajectory - estimated_trajectory, axis=1)
        errors_2d = np.linalg.norm(true_trajectory[:, :2] - estimated_trajectory[:, :2], axis=1)
        mean_error_3d = np.mean(errors_3d)
        mean_error_2d = np.mean(errors_2d)

        fig.update_layout(
            title=f'3D Indoor Localization Trajectory<br>Mean 3D Error: {mean_error_3d:.2f} m | Mean 2D Error: {mean_error_2d:.2f} m',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            showlegend=True,
            hovermode='closest',
            width=1200,
            height=900
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"轨迹图已保存到: {save_path}")

        fig.show()

    def plot_error_cdf(self, errors: np.ndarray, save_path: str = None):
        """
        绘制定位误差CDF曲线

        Args:
            errors: 误差数组
            save_path: 保存路径
        """
        # 排序误差
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

        fig = go.Figure()

        # 绘制CDF曲线
        fig.add_trace(go.Scatter(
            x=sorted_errors,
            y=cdf,
            mode='lines',
            line=dict(color='blue', width=3),
            name='CDF',
            hoverinfo='x+y'
        ))

        # 标记关键百分位
        percentiles = [50, 75, 90, 95]
        colors = ['green', 'orange', 'red', 'darkred']

        for p, color in zip(percentiles, colors):
            idx = int(len(sorted_errors) * p / 100)
            error_at_p = sorted_errors[idx]

            # 垂直线
            fig.add_trace(go.Scatter(
                x=[error_at_p, error_at_p],
                y=[0, p/100],
                mode='lines',
                line=dict(color=color, width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))

            # 水平线
            fig.add_trace(go.Scatter(
                x=[0, error_at_p],
                y=[p/100, p/100],
                mode='lines',
                line=dict(color=color, width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))

            # 标注点
            fig.add_trace(go.Scatter(
                x=[error_at_p],
                y=[p/100],
                mode='markers+text',
                marker=dict(size=8, color=color),
                text=f'{p}%: {error_at_p:.2f}m',
                textposition='top right',
                showlegend=False,
                hoverinfo='text'
            ))

        fig.update_layout(
            title='Localization Error Cumulative Distribution Function (CDF)',
            xaxis_title='Localization Error (m)',
            yaxis_title='Cumulative Probability',
            hovermode='x',
            width=1000,
            height=700,
            xaxis=dict(range=[0, sorted_errors[-1] * 1.05]),
            yaxis=dict(range=[0, 1])
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"CDF图已保存到: {save_path}")

        fig.show()


if __name__ == "__main__":
    print("高性能可视化工具模块 (Plotly)")
    print("\n使用示例:")
    print("  viz = VisualizerPlotly()")
    print("  viz.plot_signal_heatmap(fingerprint_db, ap_index=0)")
    print("  viz.plot_localization_result(true_pos, estimated_pos, fingerprint_db)")
