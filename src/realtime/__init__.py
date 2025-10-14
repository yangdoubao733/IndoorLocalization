"""
实时定位模块
支持非合作目标定位 - 通用电磁信号追踪
"""

# 原始WiFi专用模块（保留兼容性）
from .signal_collector import SignalCollector, SimulatedSignalCollector, RealAPSignalCollector

# 新增通用电磁信号模块
from .em_signal_collector import (
    UniversalEMSignalCollector,
    RealEMSignalCollector,
    SignalType,
    EMTarget
)

from .device_tracker import DeviceTracker, TargetDevice

__all__ = [
    # WiFi专用（向后兼容）
    'SignalCollector',
    'SimulatedSignalCollector',
    'RealAPSignalCollector',
    # 通用电磁信号
    'UniversalEMSignalCollector',
    'RealEMSignalCollector',
    'SignalType',
    'EMTarget',
    # 设备跟踪
    'DeviceTracker',
    'TargetDevice'
]
