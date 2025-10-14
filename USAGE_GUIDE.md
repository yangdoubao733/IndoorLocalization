# 使用指南

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备SketchUp模型

将您在SketchUp中制作的室内模型导出为COLLADA (.dae)格式:

1. 在SketchUp中打开模型
2. 文件 > 导出 > 3D模型
3. 选择格式: COLLADA (.dae)
4. 保存到 `data/models/` 目录

### 3. 运行系统

**重要**: SketchUp模型默认单位为毫米(mm)，系统会自动转换为米进行计算。

#### 方式1: 图形界面（推荐）

```bash
python gui.py
```

提供清晰易用的图形界面，功能包括：
- **模型加载**: 选择模型文件，指定单位(mm/m)
- **指纹库构建**: 设置采样参数，配置AP位置
- **定位测试**: 单点定位或批量评估
- **系统配置**: 调整电磁仿真参数
- **实时日志**: 查看运行状态和结果

#### 方式2: 命令行模式

有三种运行模式:

##### 模式1: 演示模式（推荐首次使用）

```bash
python main.py --mode demo --model data/models/your_model.dae --visualize
```

这将自动完成:
- 加载模型
- 构建指纹库
- 执行定位测试
- 可视化结果

##### 模式2: 仅构建指纹库

```bash
python main.py --mode build --model data/models/your_model.dae --grid-spacing 1.0 --height 1.5 --visualize
```

参数说明:
- `--grid-spacing`: 指纹库采样点间距（米），默认1.0m
- `--height`: 采样高度（米），默认1.5m
- `--visualize`: 可视化信号分布热图

##### 模式3: 仅执行定位

```bash
# 单点定位测试
python main.py --mode locate --fingerprint data/fingerprints/fingerprint.pkl --test-position 5.0 5.0 1.5 --algorithm wknn --k 4 --visualize

# 批量定位评估（不指定test-position）
python main.py --mode locate --fingerprint data/fingerprints/fingerprint.pkl --algorithm wknn --k 4 --visualize
```

参数说明:
- `--algorithm`: 定位算法，可选 knn, wknn, probabilistic
- `--k`: K近邻数量，默认4
- `--test-position X Y Z`: 指定测试位置坐标

## 系统配置

编辑 `config.py` 文件可以调整:

### 电磁仿真参数

```python
EM_SIMULATION_CONFIG = {
    'tx_power': 20.0,           # 发射功率 (dBm)
    'tx_frequency': 2.4e9,      # 工作频率 2.4GHz
    'max_reflections': 3,       # 最大反射次数
}
```

### AP位置配置

```python
FINGERPRINT_CONFIG = {
    'ap_positions': [           # 修改为您的AP实际位置
        (5.0, 5.0, 2.5),
        (15.0, 5.0, 2.5),
        (5.0, 15.0, 2.5),
        (15.0, 15.0, 2.5),
    ],
}
```

### 定位算法配置

```python
LOCALIZATION_CONFIG = {
    'algorithm': 'wknn',        # knn, wknn, probabilistic
    'k_neighbors': 4,           # K值
}
```

## 程序化使用

您也可以在Python代码中直接使用各模块:

```python
from src.models import load_model
from src.simulation import create_ray_tracer
from src.fingerprint import build_fingerprint_database, FingerprintDatabase
from src.localization import create_localization_engine
from src.utils import Visualizer
from config import *

# 1. 加载模型（指定单位为毫米）
model = load_model('data/models/room.dae', unit='mm')

# 2. 创建射线追踪器
ray_tracer = create_ray_tracer(model, EM_SIMULATION_CONFIG)

# 3. 构建指纹库
fingerprint_db = build_fingerprint_database(model, ray_tracer, FINGERPRINT_CONFIG)
fingerprint_db.save('data/fingerprints/my_fingerprint.pkl')

# 4. 加载指纹库并定位
fingerprint_db = FingerprintDatabase.load('data/fingerprints/my_fingerprint.pkl')
localization_engine = create_localization_engine(fingerprint_db, LOCALIZATION_CONFIG)

# 5. 执行定位
measured_rssi = np.array([-45.2, -52.3, -48.1, -55.7])  # 您的RSSI测量值
result = localization_engine.locate(measured_rssi)
print(f"估计位置: {result['position']}")
print(f"置信度: {result['confidence']}")

# 6. 可视化
viz = Visualizer()
viz.plot_signal_heatmap(fingerprint_db, ap_index=0)
viz.plot_localization_result(true_pos, result['position'], fingerprint_db)
```

## 输出文件

- **指纹库**: `data/fingerprints/*.pkl` - 包含位置-RSSI映射数据
- **热图**: `results/heatmap_*.png` - 信号强度分布热图
- **定位结果**: `results/localization_result.png` - 定位结果可视化
- **误差分析**: `results/error_cdf.png` - 定位误差CDF曲线

## 技术原理

### 1. 模型加载
使用trimesh库加载SketchUp导出的3D模型，提取墙面几何信息。

### 2. 电磁仿真
基于射线追踪算法模拟WiFi信号传播:
- 自由空间路径损耗（Friis公式）
- 墙面反射和材料衰减
- 阴影衰落效应

### 3. 指纹库构建
在室内空间均匀采样，为每个采样点计算所有AP的信号强度，形成位置-RSSI映射数据库。

### 4. 定位算法

**K-NN**: 找到K个最相似的指纹点，取平均位置
**WKNN**: 加权K近邻，根据相似度加权平均
**概率法**: 基于高斯模型计算每个点的概率，加权平均

## 常见问题

**Q: 模型加载失败?**
A: 确保模型格式为.dae/.obj/.stl。如果模型单位是毫米，在GUI中选择"mm"或在代码中指定unit='mm'

**Q: 定位精度不高?**
A: 尝试减小grid_spacing（增加采样点密度），调整K值，或更换定位算法

**Q: 程序运行很慢?**
A: 增大grid_spacing减少采样点数量，或减少AP数量

**Q: 如何使用真实RSSI数据?**
A: 参考上面的程序化使用示例，直接调用localization_engine.locate(measured_rssi)

## 联系方式

如有问题或建议，欢迎在GitHub提交Issue。
