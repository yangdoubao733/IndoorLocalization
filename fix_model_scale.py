"""
修复模型单位问题 - 将毫米模型转换为米
"""
import trimesh
import numpy as np

def fix_model_scale(input_path, output_path, scale_factor=0.001):
    """
    缩放模型

    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径
        scale_factor: 缩放系数（0.001 = mm转m）
    """
    print(f"加载模型: {input_path}")
    mesh = trimesh.load(input_path, force='mesh')

    # 如果是场景，合并
    if isinstance(mesh, trimesh.Scene):
        geometries = []
        for name, geom in mesh.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                geometries.append(geom)
        if geometries:
            mesh = trimesh.util.concatenate(geometries)

    print(f"原始边界: {mesh.bounds}")
    print(f"原始尺寸: X={mesh.bounds[1][0] - mesh.bounds[0][0]:.2f}, "
          f"Y={mesh.bounds[1][1] - mesh.bounds[0][1]:.2f}, "
          f"Z={mesh.bounds[1][2] - mesh.bounds[0][2]:.2f}")

    # 缩放
    print(f"\n应用缩放: {scale_factor}")
    mesh.apply_scale(scale_factor)

    print(f"缩放后边界: {mesh.bounds}")
    print(f"缩放后尺寸: X={mesh.bounds[1][0] - mesh.bounds[0][0]:.2f}m, "
          f"Y={mesh.bounds[1][1] - mesh.bounds[0][1]:.2f}m, "
          f"Z={mesh.bounds[1][2] - mesh.bounds[0][2]:.2f}m")

    # 保存
    print(f"\n保存到: {output_path}")
    mesh.export(output_path)
    print("完成!")

if __name__ == "__main__":
    fix_model_scale(
        "data/models/siyuanlou.dae",
        "data/models/siyuanlou_fixed.dae",
        scale_factor=0.001  # mm转m
    )
