# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供在此代码库中工作时的指导信息。

## 项目概述

这是一个基于**几何电磁孪生（Geometric Electromagnetic Digital Twin）**的**室内定位系统**。系统从SketchUp加载3D模型，使用射线追踪进行电磁仿真，构建无线信号指纹库，实现室内目标的精确定位。

核心功能：
- **合作定位**：传统的基于指纹的定位
- **非合作定位**：被动追踪电磁信号源（WiFi、蓝牙、手机信号、RFID、ZigBee、LoRa、UWB）
- **射线追踪模式**：简化模式（快速）、高精度反射模式（精确）、多径传播模式（极高精度）

## 运行系统

### 图形界面模式（推荐）
```bash
python gui.py
```

### 命令行模式

构建指纹库：
```bash
python main.py --mode build --model data/models/your_model.dae --grid-spacing 1.0 --height 1.5
```

测试定位：
```bash
python main.py --mode locate --fingerprint data/fingerprints/fingerprint.pkl --algorithm wknn --k 4
```

单点测试：
```bash
python main.py --mode locate --fingerprint data/fingerprints/fingerprint.pkl --test-position 5.0 5.0 1.5
```

完整演示：
```bash
python main.py --mode demo --model data/models/room.dae
```

### 测试
目前没有正式的测试套件。验证功能的方法：
```bash
# 测试射线追踪模块
python src/simulation/ray_tracing.py

# 测试定位算法
python src/localization/algorithms.py

# 测试电磁信号采集
python src/realtime/em_signal_collector.py
```

## 系统架构

### 两阶段流程

**离线阶段（指纹库构建）**：
```
3D模型 → 射线追踪仿真 → 网格采样 → 指纹数据库
```

**在线阶段（实时定位）**：
```
信号测量 → 指纹匹配 → K-NN算法 → 位置估算
```

### 模块结构

```
src/
├── models/              # 3D模型加载（使用trimesh）
│   └── model_loader.py  # 处理.dae/.obj/.stl，自动单位转换（mm→m）
├── simulation/          # 电磁仿真
│   ├── ray_tracing.py   # 射线追踪、路径损耗模型、反射追踪
│   └── multipath_tracing.py  # 多径传播追踪、Fibonacci球面采样
├── fingerprint/         # 指纹数据库
│   └── builder.py       # 批量向量化RSSI计算
├── localization/        # 定位算法
│   └── algorithms.py    # K-NN、WKNN、概率定位
├── realtime/            # 非合作定位
│   ├── em_signal_collector.py  # 多信号类型支持
│   └── device_tracker.py       # 实时设备追踪
└── utils/               # 可视化和工具函数
```

### 配置系统

所有系统参数集中在 `config.py` 中：

- **EM_SIMULATION_CONFIG**：射线追踪参数（频率、功率、最大反射次数）
- **FINGERPRINT_CONFIG**：网格间距、AP位置、采样高度
- **LOCALIZATION_CONFIG**：算法选择（knn/wknn/probabilistic）、K值
- **REALTIME_TRACKING_CONFIG**：信号采集模式（模拟/真实）
- **EM_SIGNAL_TRACKING_CONFIG**：信号类型参数（WiFi、蓝牙等）

### 射线追踪模式

系统支持三种射线追踪模式：

**简化模式**（默认）：
- 固定反射损耗（每次反射-5dB）
- 快速计算（121个点约1秒）
- 精度：1-2米

**高精度模式**：
- 真实镜面反射追踪（最多10次反弹）
- 材料特定的反射系数
- 考虑入射角的Fresnel反射效应
- 较慢计算（约20-30秒）
- 精度：0.5-1.2米

**多径传播模式**（最真实）：
- 全方向发射射线（Fibonacci球面采样）
- 追踪所有能到达接收点的路径
- 线性功率叠加（非dB相加）
- 最慢计算（约1-5分钟）
- 精度：<0.5米（极高精度）

通过配置启用：
```python
# 高精度模式
config = {
    'high_precision_mode': True,
    'max_reflections': 3,
    'default_material': 'concrete',
    'custom_materials': {
        'my_wall': {
            'reflection_coefficient': 0.35,  # 反射系数 0-1
            'absorption_db': 12.0            # 吸收损耗 dB
        }
    }
}

# 多径传播模式
config = {
    'multipath_enabled': True,
    'num_rays': 360,              # 射线数量
    'rx_tolerance': 0.3,          # 接收容差（米）
    'power_threshold_dbm': -100.0,  # 功率阈值
    'high_precision_mode': True,  # 需同时启用高精度
    'max_reflections': 3,
}
```

详见 `material_config_example.py`、`高精度反射模式使用指南.md` 和 `多径传播模式使用指南.md`。

## 关键设计模式

### 1. 模型单位处理

**问题**：SketchUp默认导出为毫米，但计算需要米。

**解决方案**：`model_loader.py` 实现了智能单位检测：
1. 解析COLLADA文件头中的单位标签
2. 退回到基于尺寸的启发式判断
3. 自动应用缩放变换（如 mm→m 使用0.001）

始终使用 `load_model()` 自动处理：
```python
model = load_model('path/to/model.dae')  # 单位自动转换为米
```

### 2. 批量向量化

**关键性能优化**：系统使用向量化批处理进行射线追踪。

传统方法（慢）：
```python
for rx_point in sampling_points:
    for ap in ap_positions:
        rssi = simulate_signal(ap, rx_point)  # 121×4=484次调用
```

优化方法（快100倍）：
```python
rssi_matrix = simulate_signal_batch(ap_positions, sampling_points)  # 单次调用
```

在 `ray_tracing.py:simulate_signal_batch()` 中的实现：
- 预先准备所有射线的起点/方向
- 单次trimesh求交调用处理所有射线
- 一次操作返回 (N, M) RSSI矩阵

### 3. 指纹数据库结构

数据库存储位置到RSSI的映射：
```python
fingerprints = {
    (x, y, z): [rssi_ap1, rssi_ap2, ..., rssi_apM],
    ...
}
```

关键方法：
- `add_fingerprint(position, rssi_values)`：添加单个条目
- `get_all_fingerprints()`：返回 (positions, rssi_matrix) 数组供ML算法使用
- `save(filepath)`：Pickle序列化
- `load(filepath)`：静态方法恢复数据库

### 4. 定位算法

三种算法配合自动K值选择：

**自适应K值选择**：
```python
K = min(max(8, sqrt(N)), 20)  # N = 指纹点数量
```
- 最小值：8（稳定性）
- 推荐值：√N（平衡）
- 最大值：20（避免过度平滑）

**K-NN**：K个最近邻的简单平均
**WKNN**：按距离倒数加权（精度更高）
**概率定位**：高斯概率模型（适合稀疏数据库）

所有算法继承自 `FingerprintLocalization` 基类，实现 `localize(measured_rssi)` 方法。

### 5. 非合作信号采集

**双重架构**：

`UniversalEMSignalCollector`（模拟）：
- 用于无硬件测试
- 使用路径损耗模型或指纹库查询
- 支持7种信号类型（WiFi、蓝牙、手机、RFID、ZigBee、LoRa、UWB）

`RealEMSignalCollector`（真实）：
- 通过UDP/TCP连接真实接收器
- 基于JSON的通信协议
- 命令：`get_rssi`、`scan_targets`

两者实现相同接口，可供 `DeviceTracker` 互换使用。

### 6. 射线-三角形求交

使用 **trimesh.ray.intersects_location()**：
```python
locations, index_ray, index_tri = mesh.ray.intersects_location(
    ray_origins,     # (N, 3) 数组
    ray_directions   # (N, 3) 数组
)
# 返回：交点位置、射线索引、三角形索引
```

系统将其封装在 `model.ray_intersect()` 中以保持一致性。

**高精度反射** 扩展功能：
- `get_surface_normal(tri_index)`：提取面片法向量
- `calculate_reflection_direction(incident, normal)`：镜面反射定律
- `trace_ray_with_reflections()`：递归反弹追踪

## 重要约束

### 模型要求

- **支持格式**：.dae（COLLADA）、.obj（Wavefront）、.stl
- **单位假设**：模型默认为毫米，除非另有检测
- **坐标系**：右手系，Z轴向上（SketchUp惯例）
- **网格质量**：必须是封闭网格以确保射线求交准确

### AP布置规则

- 最少：3D定位需4个AP（2D定位需3个）
- 推荐：均匀分布，避免共线
- 高度多样性提高Z轴精度
- 10m×10m房间示例：
  ```python
  [(1, 1, 2.5), (9, 1, 2.5), (1, 9, 2.5), (9, 9, 2.5)]  # 四个角
  ```

### 指纹密度 vs 性能

| 网格间距 | 点数（10×10m） | 构建时间 | 精度   |
|----------|----------------|----------|--------|
| 2.0m     | 36             | ~1秒     | ~2.5m  |
| 1.0m     | 121            | ~4秒     | ~1.2m  |
| 0.5m     | 441            | ~15秒    | ~0.6m  |

构建时间与定位精度之间的权衡。

## 文件路径和数据流

```
data/
├── models/           # 输入：SketchUp导出文件（.dae、.obj）
└── fingerprints/     # 输出：构建的数据库（.pkl）

results/              # 输出：可视化结果（PNG/HTML）
```

典型工作流：
1. 将SketchUp模型导出到 `data/models/`
2. 运行 `python main.py --mode build` → 生成 `data/fingerprints/fingerprint_*.pkl`
3. 运行 `python main.py --mode locate` → 使用.pkl文件
4. 可视化结果保存到 `results/`

## 材料配置

高精度模式的材料属性定义：
```python
{
    'material_name': {
        'reflection_coefficient': 0.0-1.0,  # 反射能量比例
        'absorption_db': float              # 穿透损耗（dB）
    }
}
```

内置材料：concrete（混凝土）、brick（砖）、wood（木材）、glass（玻璃）、metal（金属）、drywall（石膏板）

参考值（2.4 GHz）：
- **混凝土**：0.3反射系数，15 dB吸收
- **玻璃**：0.7反射系数，3 dB吸收
- **金属**：0.95反射系数，30 dB吸收

详见 `material_config_example.py` 的预设配置（办公室、地下室、仓库、住宅）。

## 常见问题

### 1. 模型尺度问题
如果定位结果离谱（如公里级），说明模型单位错误。检查控制台输出：
```
检测到单位: 毫米 (mm)，尺寸: 10000.00
应用单位转换: 毫米 -> 米 (缩放系数: 0.001)
```

### 2. AP位置不对齐
`config.py` 中的AP位置必须匹配模型坐标系。如果结果镜像/偏移，验证AP坐标。

### 3. 空指纹数据库
如果 `simulate_signal_batch()` 返回全是NaN或-100 dBm：
- 检查AP位置是否在模型边界内
- 验证射线追踪未被模型几何阻挡
- 先用 `high_precision_mode: False` 测试

### 4. 高精度模式太慢
对于>1000个采样点的高精度模式：
- 将 `max_reflections` 从3降到2
- 将 `grid_spacing` 从1.0增到2.0
- 或先用简化模式构建，再对关键区域使用高精度重新计算

## 关键文档文件

- **算法原理详解.md**：完整算法理论
- **射线追踪算法详解.md**：射线追踪深入讲解和示例
- **高精度反射模式使用指南.md**：高精度模式教程
- **多径传播模式使用指南.md**：多径追踪完整说明
- **非合作定位使用说明.md**：非合作定位指南
- **material_config_example.py**：预配置的材料设置

## 扩展系统

### 添加新信号类型

编辑 `config.py`：
```python
EM_SIGNAL_TRACKING_CONFIG = {
    'signal_types': {
        'NewSignal': {
            'frequency': 5.0e9,      # Hz
            'tx_power': 15.0,        # dBm
            'path_loss_exponent': 2.2,
            'description': '你的描述'
        }
    }
}
```

### 添加新定位算法

1. 在 `src/localization/algorithms.py` 中创建类
2. 继承自 `FingerprintLocalization`
3. 实现 `localize(measured_rssi)` 方法
4. 添加到 `LocalizationEngine.__init__()` 的switch语句

### 自定义材料属性

通过配置传递：
```python
config = {
    'custom_materials': {
        'my_material': {
            'reflection_coefficient': 0.4,
            'absorption_db': 10.0
        }
    },
    'material_mapping': {
        0: 'my_material',    # 三角形0使用此材料
        1: 'my_material',
        # ...
    }
}
```

## 性能考虑

- **指纹库构建**：O(N×M)，其中N=采样点数，M=AP数
- **批量射线追踪**：因向量化比循环快100倍
- **定位**：所有算法均为O(N)（线性扫描指纹库）
- **高精度反射**：O(N×M×R)，其中R=最大反射次数

对于大空间（>100m²）：
1. 初始使用粗网格（2米间距）
2. 仅在需要高精度的区域构建细网格
3. 考虑并行处理（未来增强）

## 版本兼容性

- Python：3.7+
- 关键依赖：numpy、trimesh、scipy、tkinter
- 已在Windows上测试（路径同时支持 `/` 和 `\`）
- 3D模型：SketchUp 2017+（COLLADA 1.4.1）

## GUI vs CLI 权衡

**GUI (`gui.py`)**：
- 对非技术用户友好
- 实时进度可视化
- 交互式参数调整
- 因UI更新较慢
- 设置存储在 `gui_settings.json`

**CLI (`main.py`)**：
- 执行更快（无UI开销）
- 可脚本化和自动化
- 更适合批处理
- 通过argparse直接访问所有参数

根据用例选择：探索用GUI，生产/自动化用CLI。

# 开发规范

- 所有文档与介绍都应使用中文
- 每进行一次更改都要保存到git同时推送到github，保存到git时应该撰写简短的说明
- 每次进行更改时不应该忘记更改gui
- 更改代码后不需要进行测试
