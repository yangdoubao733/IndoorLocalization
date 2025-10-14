"""
基于几何电磁孪生的室内非合作目标定位系统
主程序入口
"""

import argparse
import numpy as np
import os
from datetime import datetime

# 导入自定义模块
from src.models import load_model
from src.simulation import create_ray_tracer
from src.fingerprint import FingerprintDatabase, build_fingerprint_database
from src.localization import create_localization_engine
from src.utils import Visualizer

# 导入配置
from config import (
    EM_SIMULATION_CONFIG,
    FINGERPRINT_CONFIG,
    LOCALIZATION_CONFIG,
    PATHS
)


def build_mode(args):
    """构建指纹库模式"""
    print("=" * 60)
    print("模式: 构建指纹库")
    print("=" * 60)

    # 1. 加载模型
    print(f"\n[1/4] 加载3D模型: {args.model}")
    model = load_model(args.model)

    # 2. 创建射线追踪器
    print(f"\n[2/4] 初始化电磁仿真引擎")
    ray_tracer = create_ray_tracer(model, EM_SIMULATION_CONFIG)

    # 3. 构建指纹库
    print(f"\n[3/4] 构建指纹库")
    print(f"  网格间距: {args.grid_spacing} m")
    print(f"  采样高度: {args.height} m")
    print(f"  AP数量: {len(FINGERPRINT_CONFIG['ap_positions'])}")

    # 更新配置
    config = FINGERPRINT_CONFIG.copy()
    config['grid_spacing'] = args.grid_spacing
    config['height'] = args.height

    fingerprint_db = build_fingerprint_database(model, ray_tracer, config)

    # 4. 保存指纹库
    print(f"\n[4/4] 保存指纹库")
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(PATHS['fingerprints'], f'fingerprint_{timestamp}.pkl')
    else:
        output_path = args.output

    fingerprint_db.save(output_path)

    # 5. 可视化（可选）
    if args.visualize:
        print(f"\n[可选] 可视化信号分布")
        viz = Visualizer()
        viz.plot_all_aps_heatmap(
            fingerprint_db,
            save_path=os.path.join(PATHS['results'], 'heatmap_all_aps.png')
        )

    print("\n" + "=" * 60)
    print("指纹库构建完成!")
    print(f"输出文件: {output_path}")
    print("=" * 60)


def locate_mode(args):
    """定位模式"""
    print("=" * 60)
    print("模式: 室内定位")
    print("=" * 60)

    # 1. 加载指纹库
    print(f"\n[1/3] 加载指纹库: {args.fingerprint}")
    fingerprint_db = FingerprintDatabase.load(args.fingerprint)

    # 2. 创建定位引擎
    print(f"\n[2/3] 初始化定位引擎")
    print(f"  算法: {args.algorithm}")
    print(f"  K值: {args.k}")

    config = LOCALIZATION_CONFIG.copy()
    config['algorithm'] = args.algorithm
    config['k_neighbors'] = args.k

    localization_engine = create_localization_engine(fingerprint_db, config)

    # 3. 执行定位
    print(f"\n[3/3] 执行定位")

    if args.test_position is not None:
        # 单点定位测试
        test_pos = np.array(args.test_position)
        print(f"  测试位置: {test_pos}")

        # 模拟测量RSSI（从指纹库获取）
        measured_rssi = fingerprint_db.get_fingerprint(tuple(test_pos))

        if measured_rssi is None:
            # 如果不在指纹库中，使用最近点
            positions, rssi_matrix = fingerprint_db.get_all_fingerprints()
            distances = np.linalg.norm(positions - test_pos, axis=1)
            nearest_idx = np.argmin(distances)
            measured_rssi = rssi_matrix[nearest_idx]
            print(f"  (使用最近指纹点，距离 {distances[nearest_idx]:.2f}m)")

        # 定位
        result = localization_engine.locate(measured_rssi)

        print(f"\n定位结果:")
        print(f"  估计位置: [{result['position'][0]:.2f}, {result['position'][1]:.2f}, {result['position'][2]:.2f}]")
        print(f"  置信度: {result['confidence']:.3f}")

        # 计算误差
        error = np.linalg.norm(test_pos[:2] - result['position'][:2])
        print(f"  定位误差: {error:.2f} m")

        # 可视化
        if args.visualize:
            viz = Visualizer()
            viz.plot_localization_result(
                test_pos,
                result['position'],
                fingerprint_db,
                save_path=os.path.join(PATHS['results'], 'localization_result.png')
            )

    else:
        # 批量定位评估
        print("  执行批量定位评估...")

        positions, rssi_matrix = fingerprint_db.get_all_fingerprints()

        # 使用部分数据作为测试集
        test_ratio = 0.2
        num_test = int(len(positions) * test_ratio)
        test_indices = np.random.choice(len(positions), num_test, replace=False)

        test_positions = positions[test_indices]
        test_rssi = rssi_matrix[test_indices]

        # 评估
        eval_result = localization_engine.evaluate_accuracy(test_positions, test_rssi)

        # 可视化CDF
        if args.visualize:
            viz = Visualizer()
            viz.plot_error_cdf(
                eval_result['errors'],
                save_path=os.path.join(PATHS['results'], 'error_cdf.png')
            )

    print("\n" + "=" * 60)
    print("定位完成!")
    print("=" * 60)


def demo_mode(args):
    """演示模式 - 完整流程"""
    print("=" * 60)
    print("模式: 演示完整流程")
    print("=" * 60)

    # 检查是否有模型文件
    if not os.path.exists(args.model):
        print(f"\n错误: 模型文件不存在: {args.model}")
        print("\n请将SketchUp导出的模型文件 (.dae 或 .obj) 放置到 data/models/ 目录")
        print("导出方法: 在SketchUp中选择 文件 > 导出 > 3D模型 > COLLADA (.dae)")
        return

    # 构建指纹库
    print("\n" + "=" * 60)
    print("步骤 1: 构建指纹库")
    print("=" * 60)
    args_build = argparse.Namespace(
        model=args.model,
        grid_spacing=1.0,
        height=1.5,
        output='data/fingerprints/demo_fingerprint.pkl',
        visualize=True
    )
    build_mode(args_build)

    # 执行定位
    print("\n" + "=" * 60)
    print("步骤 2: 测试定位")
    print("=" * 60)
    args_locate = argparse.Namespace(
        fingerprint='data/fingerprints/demo_fingerprint.pkl',
        algorithm='wknn',
        k=4,
        test_position=None,  # 批量评估
        visualize=True
    )
    locate_mode(args_locate)

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='基于几何电磁孪生的室内非合作目标定位系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 构建指纹库
  python main.py --mode build --model data/models/room.dae

  # 执行定位
  python main.py --mode locate --fingerprint data/fingerprints/fingerprint.pkl --test-position 5.0 5.0 1.5

  # 运行演示
  python main.py --mode demo --model data/models/room.dae
        """
    )

    parser.add_argument('--mode', type=str, choices=['build', 'locate', 'demo'],
                        default='demo', help='运行模式')

    # 构建模式参数
    parser.add_argument('--model', type=str, default='data/models/indoor_scene.dae',
                        help='3D模型文件路径 (.dae, .obj, .stl)')
    parser.add_argument('--grid-spacing', type=float, default=1.0,
                        help='指纹库网格间距 (米)')
    parser.add_argument('--height', type=float, default=1.5,
                        help='采样高度 (米)')
    parser.add_argument('--output', type=str, default=None,
                        help='指纹库输出路径')

    # 定位模式参数
    parser.add_argument('--fingerprint', type=str, default='data/fingerprints/fingerprint.pkl',
                        help='指纹库文件路径')
    parser.add_argument('--algorithm', type=str, choices=['knn', 'wknn', 'probabilistic'],
                        default='wknn', help='定位算法')
    parser.add_argument('--k', type=int, default=4,
                        help='K近邻数量')
    parser.add_argument('--test-position', type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'), help='测试位置坐标')

    # 通用参数
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果')

    args = parser.parse_args()

    # 根据模式执行
    if args.mode == 'build':
        build_mode(args)
    elif args.mode == 'locate':
        locate_mode(args)
    elif args.mode == 'demo':
        demo_mode(args)


if __name__ == "__main__":
    main()
