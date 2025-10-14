"""
测试EMTarget类的修复
"""
import numpy as np
from src.realtime.em_signal_collector import EMTarget, SignalType

print("=" * 60)
print("测试EMTarget类")
print("=" * 60)

# 测试1: 使用mac参数
print("\n测试1: 使用mac参数创建WiFi设备")
try:
    target1 = EMTarget(
        mac="AA:BB:CC:DD:EE:FF",
        signal_type="WiFi",
        position=np.array([5.0, 5.0, 1.5]),
        frequency=2.4e9,
        tx_power=20.0,
        path_loss_exponent=2.0
    )
    print(f"✓ 成功创建: {target1}")
    print(f"  identifier: {target1.identifier}")
    print(f"  mac: {target1.mac}")
    print(f"  frequency: {target1.frequency}")
    print(f"  tx_power: {target1.tx_power}")
    print(f"  path_loss_exponent: {target1.path_loss_exponent}")
except Exception as e:
    print(f"✗ 失败: {e}")

# 测试2: 使用identifier参数
print("\n测试2: 使用identifier参数创建蓝牙设备")
try:
    target2 = EMTarget(
        identifier="12:34:56:78:90:AB",
        signal_type="Bluetooth",
        position=np.array([10.0, 10.0, 1.2])
    )
    print(f"✓ 成功创建: {target2}")
    print(f"  identifier: {target2.identifier}")
    print(f"  mac: {target2.mac}")
    print(f"  frequency: {target2.frequency} (自动填充)")
    print(f"  tx_power: {target2.tx_power} (自动填充)")
    print(f"  path_loss_exponent: {target2.path_loss_exponent} (自动填充)")
except Exception as e:
    print(f"✗ 失败: {e}")

# 测试3: 使用默认参数
print("\n测试3: 使用信号类型默认参数创建RFID设备")
try:
    target3 = EMTarget(
        mac="RFID-TAG-001",
        signal_type="RFID",
        position=np.array([7.0, 7.0, 1.0])
    )
    print(f"✓ 成功创建: {target3}")
    print(f"  frequency: {target3.frequency/1e6:.0f} MHz (自动填充)")
    print(f"  tx_power: {target3.tx_power} dBm (自动填充)")
    print(f"  path_loss_exponent: {target3.path_loss_exponent} (自动填充)")
except Exception as e:
    print(f"✗ 失败: {e}")

# 测试4: 测试所有信号类型
print("\n测试4: 测试所有支持的信号类型")
signal_types = ["WiFi", "Bluetooth", "Cellular", "RFID", "ZigBee", "LoRa", "UWB"]
for i, sig_type in enumerate(signal_types):
    try:
        target = EMTarget(
            mac=f"DEVICE-{i:02d}",
            signal_type=sig_type,
            position=np.array([float(i), float(i), 1.5])
        )
        print(f"✓ {sig_type:10s}: {target.frequency/1e9:.2f} GHz, {target.tx_power:6.1f} dBm, n={target.path_loss_exponent}")
    except Exception as e:
        print(f"✗ {sig_type:10s}: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
