"""
系统配置文件
定义电磁仿真、指纹库构建和定位算法的参数
"""

# 电磁仿真参数
EM_SIMULATION_CONFIG = {
    # 发射机参数
    'tx_power': 20.0,           # 发射功率 (dBm)
    'tx_frequency': 2.4e9,      # 工作频率 (Hz) - 2.4GHz WiFi
    'tx_antenna_gain': 2.0,     # 天线增益 (dBi)

    # 射线追踪参数
    'max_reflections': 3,       # 最大反射次数
    'max_diffractions': 1,      # 最大衍射次数
    'ray_resolution': 1.0,      # 射线角度分辨率 (度)

    # 材料属性 (相对介电常数, 电导率)
    'materials': {
        'concrete': {'epsilon_r': 4.5, 'sigma': 0.02},
        'brick': {'epsilon_r': 4.0, 'sigma': 0.01},
        'wood': {'epsilon_r': 2.5, 'sigma': 0.001},
        'glass': {'epsilon_r': 6.0, 'sigma': 0.0001},
        'metal': {'epsilon_r': 1.0, 'sigma': 1e7},
    },

    # 路径损耗模型参数
    'path_loss_exponent': 2.0,  # 自由空间路径损耗指数
    'shadow_fading_std': 4.0,   # 阴影衰落标准差 (dB)
}

# 指纹库构建参数
FINGERPRINT_CONFIG = {
    'grid_spacing': 1.0,        # 采样点间隔 (米)
    'height': 1.5,              # 接收天线高度 (米)
    'num_access_points': 4,     # 接入点数量
    'ap_positions': [           # AP位置 [(x,y,z), ...]
        (5.0, 5.0, 2.5),
        (15.0, 5.0, 2.5),
        (5.0, 15.0, 2.5),
        (15.0, 15.0, 2.5),
    ],
}

# 定位算法参数
LOCALIZATION_CONFIG = {
    'algorithm': 'wknn',        # 定位算法: 'knn', 'wknn', 'probabilistic'
    'k_neighbors': 4,           # K近邻数量
    'distance_metric': 'euclidean',  # 距离度量: 'euclidean', 'manhattan'
}

# 可视化参数
VISUALIZATION_CONFIG = {
    'show_rays': True,          # 显示射线路径
    'show_heatmap': True,       # 显示信号强度热图
    'show_trajectory': True,    # 显示定位轨迹
    'colormap': 'viridis',      # 颜色映射
}

# 非合作定位配置
REALTIME_TRACKING_CONFIG = {
    # 信号采集模式: 'simulated' 或 'real'
    'mode': 'simulated',

    # 真实AP配置（仅real模式使用）
    'ap_addresses': [
        '192.168.1.101',
        '192.168.1.102',
        '192.168.1.103',
        '192.168.1.104'
    ],
    'ap_port': 9999,
    'protocol': 'udp',  # 'udp' 或 'tcp'

    # 跟踪参数
    'update_interval': 1.0,  # 更新间隔（秒）
    'device_timeout': 30.0,  # 设备超时时间（秒）
    'auto_discover': True,   # 是否自动发现新设备
}

# 通用电磁信号跟踪配置
EM_SIGNAL_TRACKING_CONFIG = {
    # 支持的信号类型及其默认参数
    'signal_types': {
        'WiFi': {
            'frequency': 2.4e9,      # 2.4 GHz
            'tx_power': 20.0,        # 20 dBm
            'path_loss_exponent': 2.0,
            'description': 'WiFi (2.4GHz/5GHz)'
        },
        'Bluetooth': {
            'frequency': 2.4e9,      # 2.4 GHz
            'tx_power': 4.0,         # 4 dBm
            'path_loss_exponent': 2.0,
            'description': '蓝牙 (2.4GHz)'
        },
        'Cellular': {
            'frequency': 1.8e9,      # 1.8 GHz (常用4G频段)
            'tx_power': 23.0,        # 23 dBm
            'path_loss_exponent': 2.5,
            'description': '手机信号 (700MHz-2600MHz)'
        },
        'RFID': {
            'frequency': 915e6,      # 915 MHz (UHF RFID)
            'tx_power': 30.0,        # 30 dBm
            'path_loss_exponent': 2.2,
            'description': 'RFID (13.56MHz/915MHz)'
        },
        'ZigBee': {
            'frequency': 2.4e9,      # 2.4 GHz
            'tx_power': 0.0,         # 0 dBm
            'path_loss_exponent': 2.0,
            'description': 'ZigBee (2.4GHz)'
        },
        'LoRa': {
            'frequency': 915e6,      # 915 MHz
            'tx_power': 14.0,        # 14 dBm
            'path_loss_exponent': 2.5,
            'description': 'LoRa (433MHz/868MHz/915MHz)'
        },
        'UWB': {
            'frequency': 6.5e9,      # 6.5 GHz
            'tx_power': -10.0,       # -10 dBm
            'path_loss_exponent': 1.8,
            'description': '超宽带 (3.1-10.6GHz)'
        },
    },

    # 接收机位置（与指纹库AP位置对应）
    'receiver_positions': None,  # None表示使用FINGERPRINT_CONFIG中的ap_positions

    # 环境参数
    'environment': {
        'base_shadow_fading_std': 4.0,  # 基础阴影衰落标准差 (dB)
        'reference_distance': 1.0,       # 参考距离 (m)
    }
}

# 文件路径
PATHS = {
    'models': 'data/models/',
    'fingerprints': 'data/fingerprints/',
    'results': 'results/',
}
