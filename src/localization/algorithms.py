"""
室内定位算法模块
实现基于指纹匹配的定位算法
"""

import numpy as np
from typing import Tuple, Dict, List
from scipy.spatial.distance import euclidean, cityblock
from sklearn.neighbors import NearestNeighbors


class FingerprintLocalization:
    """基于指纹匹配的定位算法基类"""

    def __init__(self, fingerprint_db, config: Dict):
        """
        初始化

        Args:
            fingerprint_db: FingerprintDatabase对象
            config: 定位配置
        """
        self.fingerprint_db = fingerprint_db
        self.config = config

        # 获取指纹库数据
        self.ref_positions, self.ref_rssi = fingerprint_db.get_all_fingerprints()

        print(f"定位算法初始化完成")
        print(f"  参考点数量: {len(self.ref_positions)}")
        print(f"  AP数量: {self.ref_rssi.shape[1]}")

    def localize(self, measured_rssi: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        定位方法（需在子类中实现）

        Args:
            measured_rssi: 测量的RSSI值 shape=(num_aps,)

        Returns:
            (estimated_position, confidence): 估计位置 (3,), 置信度
        """
        raise NotImplementedError


class KNNLocalization(FingerprintLocalization):
    """K近邻定位算法"""

    def __init__(self, fingerprint_db, config: Dict):
        super().__init__(fingerprint_db, config)

        # 自动调整K值：根据参考点数量设置合理的K值
        num_ref_points = len(self.ref_positions)
        default_k = min(max(8, int(np.sqrt(num_ref_points))), 20)  # K值在8-20之间
        self.k = config.get('k_neighbors', default_k)
        self.distance_metric = config.get('distance_metric', 'euclidean')

        # 构建KNN模型
        metric = 'euclidean' if self.distance_metric == 'euclidean' else 'manhattan'
        self.knn_model = NearestNeighbors(n_neighbors=min(self.k, num_ref_points), metric=metric)
        self.knn_model.fit(self.ref_rssi)

        print(f"K-NN算法配置: K={self.k}, metric={metric}")

    def localize(self, measured_rssi: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        K近邻定位

        Args:
            measured_rssi: 测量的RSSI值 shape=(num_aps,)

        Returns:
            (estimated_position, confidence): 估计位置 (3,), 置信度
        """
        # 找到K个最近邻
        measured_rssi = measured_rssi.reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(measured_rssi)

        # 取K个近邻位置的平均
        nearest_positions = self.ref_positions[indices[0]]
        estimated_position = np.mean(nearest_positions, axis=0)

        # 计算置信度 (距离的倒数归一化)
        avg_distance = np.mean(distances[0])
        confidence = 1.0 / (1.0 + avg_distance)

        return estimated_position, confidence


class WKNNLocalization(FingerprintLocalization):
    """加权K近邻定位算法"""

    def __init__(self, fingerprint_db, config: Dict):
        super().__init__(fingerprint_db, config)

        # 自动调整K值：根据参考点数量设置合理的K值
        num_ref_points = len(self.ref_positions)
        default_k = min(max(8, int(np.sqrt(num_ref_points))), 20)  # K值在8-20之间
        self.k = config.get('k_neighbors', default_k)
        self.distance_metric = config.get('distance_metric', 'euclidean')

        # 构建KNN模型
        metric = 'euclidean' if self.distance_metric == 'euclidean' else 'manhattan'
        self.knn_model = NearestNeighbors(n_neighbors=min(self.k, num_ref_points), metric=metric)
        self.knn_model.fit(self.ref_rssi)

        print(f"WKNN算法配置: K={self.k}, metric={metric}")

    def localize(self, measured_rssi: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        加权K近邻定位

        Args:
            measured_rssi: 测量的RSSI值 shape=(num_aps,)

        Returns:
            (estimated_position, confidence): 估计位置 (3,), 置信度
        """
        # 找到K个最近邻
        measured_rssi = measured_rssi.reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(measured_rssi)

        # 计算权重 (距离的倒数)
        distances = distances[0]
        # 避免除零
        distances = np.maximum(distances, 1e-6)
        weights = 1.0 / distances
        weights /= np.sum(weights)  # 归一化

        # 加权平均
        nearest_positions = self.ref_positions[indices[0]]
        estimated_position = np.sum(nearest_positions * weights[:, np.newaxis], axis=0)

        # 计算置信度
        avg_distance = np.mean(distances)
        confidence = 1.0 / (1.0 + avg_distance)

        return estimated_position, confidence


class ProbabilisticLocalization(FingerprintLocalization):
    """概率定位算法（基于高斯模型）"""

    def __init__(self, fingerprint_db, config: Dict):
        super().__init__(fingerprint_db, config)

        # 计算RSSI的标准差（用于高斯模型）
        self.rssi_std = np.std(self.ref_rssi, axis=0)
        self.rssi_std = np.maximum(self.rssi_std, 1.0)  # 避免过小的标准差

        print(f"概率定位算法配置: RSSI std={np.mean(self.rssi_std):.2f} dBm")

    def _gaussian_probability(self, measured: np.ndarray, reference: np.ndarray) -> float:
        """
        计算高斯概率

        Args:
            measured: 测量值
            reference: 参考值

        Returns:
            概率值
        """
        diff = measured - reference
        exponent = -0.5 * np.sum((diff / self.rssi_std) ** 2)
        prob = np.exp(exponent)

        return prob

    def localize(self, measured_rssi: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        概率定位

        Args:
            measured_rssi: 测量的RSSI值 shape=(num_aps,)

        Returns:
            (estimated_position, confidence): 估计位置 (3,), 置信度
        """
        # 计算每个参考点的概率
        probabilities = np.array([
            self._gaussian_probability(measured_rssi, ref_rssi)
            for ref_rssi in self.ref_rssi
        ])

        # 归一化概率
        probabilities /= np.sum(probabilities)

        # 加权平均
        estimated_position = np.sum(
            self.ref_positions * probabilities[:, np.newaxis],
            axis=0
        )

        # 置信度为最大概率
        confidence = np.max(probabilities)

        return estimated_position, confidence


class LocalizationEngine:
    """定位引擎"""

    def __init__(self, fingerprint_db, config: Dict):
        """
        初始化

        Args:
            fingerprint_db: FingerprintDatabase对象
            config: 定位配置
        """
        self.fingerprint_db = fingerprint_db
        self.config = config

        # 选择定位算法
        algorithm = config.get('algorithm', 'wknn')

        if algorithm == 'knn':
            self.locator = KNNLocalization(fingerprint_db, config)
        elif algorithm == 'wknn':
            self.locator = WKNNLocalization(fingerprint_db, config)
        elif algorithm == 'probabilistic':
            self.locator = ProbabilisticLocalization(fingerprint_db, config)
        else:
            raise ValueError(f"不支持的定位算法: {algorithm}")

        print(f"定位引擎已初始化，使用算法: {algorithm}")

    def locate(self, measured_rssi: np.ndarray) -> Dict:
        """
        执行定位

        Args:
            measured_rssi: 测量的RSSI值 shape=(num_aps,)

        Returns:
            定位结果字典 {position, confidence, algorithm}
        """
        position, confidence = self.locator.localize(measured_rssi)

        result = {
            'position': position,
            'confidence': confidence,
            'algorithm': self.config.get('algorithm', 'wknn'),
            'measured_rssi': measured_rssi
        }

        return result

    def evaluate_accuracy(self, test_positions: np.ndarray, test_rssi: np.ndarray, use_3d: bool = True) -> Dict:
        """
        评估定位精度

        Args:
            test_positions: 测试位置 shape=(N, 3)
            test_rssi: 测试RSSI shape=(N, num_aps)
            use_3d: 是否计算3D误差（默认True）

        Returns:
            评估结果 {mean_error, median_error, std_error, cdf}
        """
        errors_2d = []
        errors_3d = []

        for i in range(len(test_positions)):
            result = self.locate(test_rssi[i])
            estimated_pos = result['position']
            true_pos = test_positions[i]

            # 计算2D误差
            error_2d = np.linalg.norm(estimated_pos[:2] - true_pos[:2])
            errors_2d.append(error_2d)

            # 计算3D误差
            error_3d = np.linalg.norm(estimated_pos - true_pos)
            errors_3d.append(error_3d)

        errors_2d = np.array(errors_2d)
        errors_3d = np.array(errors_3d)

        # 选择使用哪种误差
        errors = errors_3d if use_3d else errors_2d

        # 计算CDF
        cdf_points = np.linspace(0, np.max(errors), 100)
        cdf_values = [np.sum(errors <= p) / len(errors) for p in cdf_points]

        result = {
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'mean_error_2d': np.mean(errors_2d),
            'mean_error_3d': np.mean(errors_3d),
            'cdf': (cdf_points, cdf_values),
            'errors': errors,
            'errors_2d': errors_2d,
            'errors_3d': errors_3d
        }

        print(f"\n定位精度评估:")
        print(f"  平均误差 (2D): {np.mean(errors_2d):.2f} m")
        print(f"  平均误差 (3D): {np.mean(errors_3d):.2f} m")
        print(f"  中位误差: {result['median_error']:.2f} m")
        print(f"  标准差: {result['std_error']:.2f} m")
        print(f"  最大误差: {result['max_error']:.2f} m")

        return result


def create_localization_engine(fingerprint_db, config: Dict) -> LocalizationEngine:
    """
    便捷函数: 创建定位引擎

    Args:
        fingerprint_db: FingerprintDatabase对象
        config: 定位配置

    Returns:
        LocalizationEngine对象
    """
    return LocalizationEngine(fingerprint_db, config)


if __name__ == "__main__":
    print("定位算法模块测试")
    print("\n使用示例:")
    print("  from src.fingerprint import FingerprintDatabase")
    print("  db = FingerprintDatabase.load('data/fingerprints/fingerprint.pkl')")
    print("  engine = create_localization_engine(db, LOCALIZATION_CONFIG)")
    print("  result = engine.locate(measured_rssi)")
    print("  print(f'估计位置: {result['position']}')")
