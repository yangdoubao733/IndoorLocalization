"""
模型诊断脚本 - 检查模型复杂度和加载时间
"""
import time
import sys

try:
    import trimesh
    import numpy as np
except ImportError as e:
    print(f"错误: 缺少依赖库 {e}")
    print("请运行: pip install trimesh numpy")
    sys.exit(1)

def diagnose_model(model_path):
    """诊断模型"""
    print(f"=" * 60)
    print(f"模型诊断: {model_path}")
    print(f"=" * 60)

    # 1. 加载模型
    print("\n[1] 加载模型...")
    start_time = time.time()
    try:
        mesh = trimesh.load(model_path, force='mesh')
    except Exception as e:
        print(f"错误: 无法加载模型 - {e}")
        return
    load_time = time.time() - start_time
    print(f"✓ 加载耗时: {load_time:.2f}秒")

    # 2. 检查场景
    if isinstance(mesh, trimesh.Scene):
        print(f"\n[2] 场景信息:")
        print(f"  几何体数量: {len(mesh.geometry)}")
        total_vertices = 0
        total_faces = 0
        for name, geom in mesh.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                print(f"  - {name}:")
                print(f"      顶点: {len(geom.vertices)}")
                print(f"      面数: {len(geom.faces)}")
                total_vertices += len(geom.vertices)
                total_faces += len(geom.faces)

        # 合并场景
        print("\n[3] 合并场景...")
        start_time = time.time()
        geometries = []
        for name, geom in mesh.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                geometries.append(geom)
        if geometries:
            mesh = trimesh.util.concatenate(geometries)
        merge_time = time.time() - start_time
        print(f"✓ 合并耗时: {merge_time:.2f}秒")

    # 3. 模型统计
    print(f"\n[4] 最终模型统计:")
    print(f"  顶点数: {len(mesh.vertices):,}")
    print(f"  面数: {len(mesh.faces):,}")
    print(f"  边界:")
    print(f"    X: [{mesh.bounds[0][0]:.2f}, {mesh.bounds[1][0]:.2f}] (长度: {mesh.bounds[1][0] - mesh.bounds[0][0]:.2f})")
    print(f"    Y: [{mesh.bounds[0][1]:.2f}, {mesh.bounds[1][1]:.2f}] (宽度: {mesh.bounds[1][1] - mesh.bounds[0][1]:.2f})")
    print(f"    Z: [{mesh.bounds[0][2]:.2f}, {mesh.bounds[1][2]:.2f}] (高度: {mesh.bounds[1][2] - mesh.bounds[0][2]:.2f})")

    # 4. 射线追踪性能测试
    print(f"\n[5] 射线追踪性能测试...")
    num_rays = 1000

    # 生成随机射线
    center = mesh.bounds.mean(axis=0)
    ray_origins = np.tile(center, (num_rays, 1))
    angles = np.linspace(0, 2*np.pi, num_rays)
    ray_directions = np.stack([
        np.cos(angles),
        np.sin(angles),
        np.zeros(num_rays)
    ], axis=1)

    # 测试射线求交
    start_time = time.time()
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions
    )
    ray_time = time.time() - start_time

    print(f"  测试射线数: {num_rays}")
    print(f"  交点数: {len(locations)}")
    print(f"  耗时: {ray_time:.3f}秒")
    print(f"  平均速度: {num_rays/ray_time:.0f} 射线/秒")

    # 5. 性能评估
    print(f"\n[6] 性能评估:")

    # 评估面数
    if len(mesh.faces) > 100000:
        print(f"  ⚠️  警告: 面数过多 ({len(mesh.faces):,})，建议简化模型")
    elif len(mesh.faces) > 50000:
        print(f"  ⚠️  注意: 面数较多 ({len(mesh.faces):,})，可能影响性能")
    else:
        print(f"  ✓ 面数适中 ({len(mesh.faces):,})")

    # 评估射线追踪速度
    rays_per_sec = num_rays / ray_time
    if rays_per_sec < 1000:
        print(f"  ⚠️  警告: 射线追踪较慢 ({rays_per_sec:.0f} 射线/秒)")
    elif rays_per_sec < 5000:
        print(f"  ⚠️  注意: 射线追踪速度一般 ({rays_per_sec:.0f} 射线/秒)")
    else:
        print(f"  ✓ 射线追踪速度良好 ({rays_per_sec:.0f} 射线/秒)")

    # 6. 预估构建时间
    print(f"\n[7] 指纹库构建时间预估:")

    # 假设不同网格配置
    configs = [
        ("1.0m 网格, 2D", 1.0, 1.5, None),
        ("0.5m 网格, 2D", 0.5, 1.5, None),
        ("1.0m 网格, 3D (5层)", 1.0, None, 5),
    ]

    for desc, spacing, height, num_z_layers in configs:
        # 计算采样点数
        x_range = mesh.bounds[1][0] - mesh.bounds[0][0]
        y_range = mesh.bounds[1][1] - mesh.bounds[0][1]

        num_x = int(np.ceil(x_range / spacing)) + 1
        num_y = int(np.ceil(y_range / spacing)) + 1

        if num_z_layers is None:
            num_points = num_x * num_y
        else:
            num_points = num_x * num_y * num_z_layers

        # 假设4个AP
        num_aps = 4
        total_rays = num_points * num_aps

        # 预估时间（考虑批量加速）
        estimated_time_old = total_rays / rays_per_sec  # 旧方法（逐个）
        estimated_time_new = total_rays / rays_per_sec / 100  # 新方法（批量，约100倍加速）

        print(f"  {desc}:")
        print(f"    采样点: {num_points:,} 个")
        print(f"    射线总数: {total_rays:,}")
        print(f"    预估时间 (旧方法): {estimated_time_old:.1f}秒 ({estimated_time_old/60:.1f}分钟)")
        print(f"    预估时间 (批量): {estimated_time_new:.1f}秒")

    print(f"\n" + "=" * 60)
    print("诊断完成!")
    print("=" * 60)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "data/models/siyuanlou.dae"

    diagnose_model(model_path)
