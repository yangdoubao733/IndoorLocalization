"""
模型加载模块
支持从SketchUp导出的COLLADA (.dae) 或 Wavefront (.obj) 格式加载3D模型
"""

import numpy as np
import trimesh
from typing import Tuple, List, Dict
import os


class IndoorModel:
    """室内环境模型类"""

    def __init__(self, model_path: str, unit: str = 'auto'):
        """
        初始化模型

        Args:
            model_path: 模型文件路径 (支持 .dae, .obj 格式)
            unit: 模型单位 ('mm' 毫米, 'm' 米, 'auto' 自动检测), 默认 'auto'
        """
        self.model_path = model_path
        self.unit = unit
        # 初始缩放系数（auto模式会在加载时动态设置）
        if unit == 'mm':
            self.scale_factor = 0.001
        elif unit == 'm':
            self.scale_factor = 1.0
        else:  # 'auto' 或其他
            self.scale_factor = 1.0  # 默认值，会在 _load_model 中更新

        self.mesh = None
        self.walls = []
        self.bounds = None
        self.materials = {}

        self._load_model()

    def _load_model(self):
        """加载3D模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        file_ext = os.path.splitext(self.model_path)[1].lower()

        if file_ext not in ['.dae', '.obj', '.stl']:
            raise ValueError(f"不支持的文件格式: {file_ext}. 请使用 .dae, .obj 或 .stl 格式")

        try:
            # 使用trimesh加载模型
            # 注意：trimesh会自动处理COLLADA文件中的单位信息
            self.mesh = trimesh.load(self.model_path, force='mesh')

            # 如果是场景，合并所有网格
            if isinstance(self.mesh, trimesh.Scene):
                geometries = []
                for name, geom in self.mesh.geometry.items():
                    if isinstance(geom, trimesh.Trimesh):
                        geometries.append(geom)
                if geometries:
                    self.mesh = trimesh.util.concatenate(geometries)
                else:
                    raise ValueError("场景中没有有效的网格数据")

            # 检查初始边界，判断是否需要单位转换
            initial_bounds = self.mesh.bounds
            initial_size = np.max(initial_bounds[1] - initial_bounds[0])

            # 单位转换逻辑
            # trimesh 有时无法正确处理 COLLADA 的单位，需要智能检测
            need_scale = False
            detected_unit = None

            if file_ext == '.dae':
                # COLLADA文件：优先读取文件内的单位信息
                # 先检查文件内容，而不是根据尺寸推测
                detected_from_file = False
                try:
                    with open(self.model_path, 'r', encoding='utf-8') as f:
                        content = f.read(5000)  # 读取文件头部
                        if 'name="inch"' in content or 'meter="0.0254"' in content:
                            detected_unit = '英寸 (inch)'
                            need_scale = True
                            self.scale_factor = 0.0254  # 英寸转米
                            detected_from_file = True
                        elif 'name="centimeter"' in content or 'meter="0.01"' in content:
                            detected_unit = '厘米 (cm)'
                            need_scale = True
                            self.scale_factor = 0.01
                            detected_from_file = True
                        elif 'name="millimeter"' in content or 'meter="0.001"' in content:
                            detected_unit = '毫米 (mm)'
                            need_scale = True
                            self.scale_factor = 0.001
                            detected_from_file = True
                        elif 'name="meter"' in content and 'meter="1"' in content:
                            detected_unit = '米 (m)'
                            need_scale = False
                            detected_from_file = True
                except Exception as e:
                    print(f"警告: 无法读取文件单位信息: {e}")
                    detected_from_file = False

                # 如果文件中没找到单位信息，才根据尺寸推测
                if not detected_from_file:
                    if 10 <= initial_size <= 300:
                        # 10-300米范围，可能是米（正常）
                        detected_unit = '米 (推测)'
                        need_scale = False
                    elif 1000 <= initial_size <= 30000:
                        # 1000-30000的数值，可能是毫米
                        detected_unit = '毫米 (推测)'
                        need_scale = True
                        self.scale_factor = 0.001
                    elif 100 <= initial_size <= 3000:
                        # 100-3000的数值，可能是厘米或英寸
                        detected_unit = '厘米或英寸 (推测)'
                        need_scale = True
                        self.scale_factor = 0.01  # 默认假设厘米
                    elif initial_size < 10:
                        # 小于10，可能已经是米但尺寸太小
                        detected_unit = '米 (尺寸较小)'
                        need_scale = False
                    else:
                        # 其他情况
                        detected_unit = '未知 (请手动检查)'
                        need_scale = False

                if need_scale:
                    print(f"检测到单位: {detected_unit}，尺寸: {initial_size:.2f}")
                    print(f"应用单位转换: {detected_unit} -> 米 (缩放系数: {self.scale_factor})")
                else:
                    print(f"检测到单位: {detected_unit}，尺寸: {initial_size:.2f}米")
                    print(f"使用COLLADA文件内置单位（无需转换）")
            else:
                # OBJ/STL文件：总是应用用户指定的单位
                need_scale = (self.scale_factor != 1.0)

            # 执行缩放
            if need_scale and self.scale_factor != 1.0:
                self.mesh.apply_scale(self.scale_factor)
            elif file_ext != '.dae' and self.scale_factor != 1.0:
                self.mesh.apply_scale(self.scale_factor)
                print(f"模型单位转换: {self.unit} -> m (缩放系数: {self.scale_factor})")

            # 计算边界
            self.bounds = self.mesh.bounds

            print(f"模型加载成功: {self.model_path}")
            print(f"  单位: {self.unit}")
            print(f"  顶点数: {len(self.mesh.vertices)}")
            print(f"  面数: {len(self.mesh.faces)}")
            print(f"  边界 (米): X[{self.bounds[0][0]:.2f}, {self.bounds[1][0]:.2f}], "
                  f"Y[{self.bounds[0][1]:.2f}, {self.bounds[1][1]:.2f}], "
                  f"Z[{self.bounds[0][2]:.2f}, {self.bounds[1][2]:.2f}]")

        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")

    def extract_walls(self, vertical_threshold: float = 0.7) -> List[np.ndarray]:
        """
        提取垂直墙面

        Args:
            vertical_threshold: 垂直判断阈值 (法向量Z分量的绝对值小于此值认为是墙面)

        Returns:
            墙面三角形列表
        """
        if self.mesh is None:
            raise RuntimeError("模型未加载")

        walls = []

        # 获取所有面的法向量
        face_normals = self.mesh.face_normals

        # 筛选垂直面 (Z分量接近0)
        for i, normal in enumerate(face_normals):
            if abs(normal[2]) < vertical_threshold:  # 垂直墙面
                # 获取该面的三个顶点
                face_vertices = self.mesh.vertices[self.mesh.faces[i]]
                walls.append(face_vertices)

        self.walls = walls
        print(f"提取墙面数量: {len(walls)}")

        return walls

    def get_floor_bounds(self) -> Tuple[float, float, float, float]:
        """
        获取地面边界

        Returns:
            (x_min, x_max, y_min, y_max)
        """
        if self.bounds is None:
            raise RuntimeError("模型未加载")

        return (
            self.bounds[0][0], self.bounds[1][0],
            self.bounds[0][1], self.bounds[1][1]
        )

    def generate_sampling_grid(self, spacing: float = 1.0, height: float = None,
                              z_min: float = None, z_max: float = None, z_spacing: float = None) -> np.ndarray:
        """
        生成采样网格点 (支持2D或3D网格)

        Args:
            spacing: XY平面网格间距 (米)
            height: 固定采样高度 (米)，用于2D网格。如果指定，则生成单一高度的2D网格
            z_min: Z方向最小值 (米)，用于3D网格
            z_max: Z方向最大值 (米)，用于3D网格
            z_spacing: Z方向网格间距 (米)，用于3D网格。如果为None，使用spacing的值

        Returns:
            采样点数组 shape=(N, 3)
        """
        x_min, x_max, y_min, y_max = self.get_floor_bounds()

        # 生成XY平面网格 - 使用linspace确保包含边界点
        # 计算需要的点数
        num_x = int(np.ceil((x_max - x_min) / spacing)) + 1
        num_y = int(np.ceil((y_max - y_min) / spacing)) + 1
        x_coords = np.linspace(x_min, x_max, num_x)
        y_coords = np.linspace(y_min, y_max, num_y)

        # 判断是2D还是3D网格
        if height is not None:
            # 2D网格：固定高度
            xx, yy = np.meshgrid(x_coords, y_coords)
            zz = np.full_like(xx, height)
            points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
            print(f"生成2D采样网格: {len(points)} 个点 (XY间距 {spacing}m, 固定高度 {height}m)")

        else:
            # 3D网格：多层高度
            if z_min is None or z_max is None:
                # 如果没有指定Z范围，使用模型边界
                z_min = self.bounds[0][2] if z_min is None else z_min
                z_max = self.bounds[1][2] if z_max is None else z_max

            if z_spacing is None:
                z_spacing = spacing  # 默认Z间距与XY间距相同

            # 使用linspace确保包含边界点
            num_z = int(np.ceil((z_max - z_min) / z_spacing)) + 1
            z_coords = np.linspace(z_min, z_max, num_z)

            # 生成3D网格
            xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords)
            points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
            print(f"生成3D采样网格: {len(points)} 个点")
            print(f"  XY间距: {spacing}m, Z间距: {z_spacing}m")
            print(f"  X: [{x_min:.2f}, {x_max:.2f}], Y: [{y_min:.2f}, {y_max:.2f}], Z: [{z_min:.2f}, {z_max:.2f}]")
            print(f"  网格维度: {len(x_coords)} x {len(y_coords)} x {len(z_coords)}")

        return points

    def ray_intersect(self, ray_origins: np.ndarray, ray_directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        射线与模型求交

        Args:
            ray_origins: 射线起点 shape=(N, 3)
            ray_directions: 射线方向 shape=(N, 3)

        Returns:
            (locations, index_ray, index_tri): 交点位置, 射线索引, 三角形索引
        """
        if self.mesh is None:
            raise RuntimeError("模型未加载")

        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        return locations, index_ray, index_tri

    def visualize(self, show: bool = True):
        """可视化模型"""
        if self.mesh is None:
            raise RuntimeError("模型未加载")

        scene = trimesh.Scene(self.mesh)

        if show:
            scene.show()

        return scene


def load_model(model_path: str, unit: str = 'auto') -> IndoorModel:
    """
    便捷函数: 加载室内模型

    Args:
        model_path: 模型文件路径
        unit: 模型单位 ('mm' 毫米, 'm' 米, 'auto' 自动检测), 默认 'auto'

    Returns:
        IndoorModel对象
    """
    return IndoorModel(model_path, unit)


if __name__ == "__main__":
    # 测试代码
    print("模型加载模块测试")
    print("请确保在 data/models/ 目录下有模型文件")
    print("\n使用示例:")
    print("  model = load_model('data/models/indoor_scene.dae')")
    print("  walls = model.extract_walls()")
    print("  grid = model.generate_sampling_grid(spacing=1.0)")
    print("  model.visualize()")
