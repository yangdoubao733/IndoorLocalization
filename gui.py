"""
基于几何电磁孪生的室内定位系统 - GUI界面
使用tkinter创建图形用户界面
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import numpy as np
import os
import sys
import json
from datetime import datetime

# 导入自定义模块
from src.models import load_model
from src.simulation import create_ray_tracer
from src.fingerprint import FingerprintDatabase, build_fingerprint_database
from src.localization import create_localization_engine
from src.utils import Visualizer, VisualizerPlotly
from src.realtime import (
    SimulatedSignalCollector, RealAPSignalCollector,
    UniversalEMSignalCollector, RealEMSignalCollector,
    SignalType, EMTarget, DeviceTracker
)
from config import EM_SIMULATION_CONFIG, FINGERPRINT_CONFIG, LOCALIZATION_CONFIG, PATHS

# 设置文件路径
SETTINGS_FILE = 'gui_settings.json'


class IndoorLocalizationGUI:
    """室内定位系统GUI"""

    def __init__(self, root):
        self.root = root
        self.root.title("室内非合作目标定位系统")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # 系统状态
        self.model = None
        self.ray_tracer = None
        self.fingerprint_db = None
        self.localization_engine = None
        self.model_path = None
        self.fingerprint_path = None

        # 非合作定位状态
        self.signal_collector = None
        self.device_tracker = None
        self.tracking_active = False

        # 高精度模式配置
        self._current_preset_config = None

        # 创建界面
        self._create_widgets()

        # 加载保存的设置
        self._load_settings()

    def _create_widgets(self):
        """创建界面组件"""

        # 创建主标题
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=10, pady=10)

        title_label = ttk.Label(
            title_frame,
            text="基于几何电磁孪生的室内定位系统",
            font=("Arial", 16, "bold")
        )
        title_label.pack()

        # 创建Notebook（选项卡）
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 选项卡0: 主页
        self.home_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.home_frame, text="主页")
        self._create_home_tab()

        # 选项卡1: 构建指纹库
        self.build_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.build_frame, text="1. 构建指纹库")
        self._create_build_tab()

        # 选项卡2: 定位测试
        self.locate_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.locate_frame, text="2. 定位测试")
        self._create_locate_tab()

        # 选项卡3: 非合作定位
        self.realtime_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.realtime_frame, text="3. 非合作定位")
        self._create_realtime_tab()

        # 选项卡4: 系统配置
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="4. 系统配置")
        self._create_config_tab()

        # 底部日志输出区域
        log_frame = ttk.LabelFrame(self.root, text="系统日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=8,
            wrap=tk.WORD,
            font=("Consolas", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 清空日志按钮
        clear_btn = ttk.Button(log_frame, text="清空日志", command=self.clear_log)
        clear_btn.pack(anchor=tk.E, pady=5)

    def _create_home_tab(self):
        """创建主页选项卡"""

        # 创建滚动区域
        canvas = tk.Canvas(self.home_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.home_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 欢迎区域
        welcome_frame = ttk.Frame(scrollable_frame)
        welcome_frame.pack(fill=tk.X, padx=20, pady=20)

        welcome_label = ttk.Label(
            welcome_frame,
            text="欢迎使用室内非合作目标定位系统",
            font=("Arial", 18, "bold"),
            foreground="#2e5090"
        )
        welcome_label.pack(pady=(0, 10))

        subtitle_label = ttk.Label(
            welcome_frame,
            text="基于几何电磁孪生技术的高精度室内定位解决方案",
            font=("Arial", 11),
            foreground="#666666"
        )
        subtitle_label.pack()

        # 系统状态区域
        status_frame = ttk.LabelFrame(scrollable_frame, text="系统状态", padding=15)
        status_frame.pack(fill=tk.X, padx=20, pady=10)

        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill=tk.X)

        # 状态指示器
        ttk.Label(status_grid, text="3D模型:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=5, padx=(0, 10))
        self.home_model_status = ttk.Label(status_grid, text="未加载", foreground="orange")
        self.home_model_status.grid(row=0, column=1, sticky=tk.W, pady=5)

        ttk.Label(status_grid, text="指纹库:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, pady=5, padx=(0, 10))
        self.home_fp_status = ttk.Label(status_grid, text="未加载", foreground="orange")
        self.home_fp_status.grid(row=1, column=1, sticky=tk.W, pady=5)

        ttk.Label(status_grid, text="定位引擎:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=5, padx=(0, 10))
        self.home_engine_status = ttk.Label(status_grid, text="未初始化", foreground="orange")
        self.home_engine_status.grid(row=2, column=1, sticky=tk.W, pady=5)

        ttk.Label(status_grid, text="跟踪系统:", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky=tk.W, pady=5, padx=(0, 10))
        self.home_tracking_status = ttk.Label(status_grid, text="未初始化", foreground="orange")
        self.home_tracking_status.grid(row=3, column=1, sticky=tk.W, pady=5)

        # 快速开始区域
        quickstart_frame = ttk.LabelFrame(scrollable_frame, text="快速开始指南", padding=15)
        quickstart_frame.pack(fill=tk.X, padx=20, pady=10)

        steps = [
            ("步骤 1", "加载3D模型", "在'构建指纹库'页面选择并加载室内环境的3D模型文件（支持.dae, .obj, .stl格式）"),
            ("步骤 2", "构建指纹库", "配置参数后点击'开始构建指纹库'，系统将自动进行电磁仿真并生成信号指纹数据"),
            ("步骤 3", "加载指纹库", "在'定位测试'页面加载已构建的指纹库文件"),
            ("步骤 4", "测试定位", "输入测试位置坐标，选择定位算法，进行单点或批量定位测试"),
            ("步骤 5", "实时跟踪", "在'非合作定位'页面初始化跟踪系统，添加设备并开始实时跟踪")
        ]

        for i, (step_num, step_title, step_desc) in enumerate(steps):
            step_frame = ttk.Frame(quickstart_frame)
            step_frame.pack(fill=tk.X, pady=5)

            # 步骤编号和标题
            step_header = ttk.Frame(step_frame)
            step_header.pack(fill=tk.X)

            step_num_label = ttk.Label(
                step_header,
                text=step_num,
                font=("Arial", 10, "bold"),
                foreground="#2e5090"
            )
            step_num_label.pack(side=tk.LEFT, padx=(0, 5))

            step_title_label = ttk.Label(
                step_header,
                text=step_title,
                font=("Arial", 10, "bold")
            )
            step_title_label.pack(side=tk.LEFT)

            # 步骤描述
            step_desc_label = ttk.Label(
                step_frame,
                text=step_desc,
                wraplength=700,
                font=("Arial", 9),
                foreground="#555555"
            )
            step_desc_label.pack(fill=tk.X, padx=(0, 0), pady=(2, 0))

            # 添加分隔线（除了最后一个步骤）
            if i < len(steps) - 1:
                ttk.Separator(quickstart_frame, orient='horizontal').pack(fill=tk.X, pady=8)

        # 功能特性区域
        features_frame = ttk.LabelFrame(scrollable_frame, text="核心功能", padding=15)
        features_frame.pack(fill=tk.X, padx=20, pady=10)

        features = [
            ("射线追踪仿真", "基于射线追踪技术的高精度电磁传播模拟"),
            ("多算法支持", "支持KNN、WKNN、概率定位等多种定位算法"),
            ("2D/3D定位", "灵活选择2D平面定位或3D空间定位模式"),
            ("实时跟踪", "支持多设备实时位置跟踪与可视化"),
            ("多信号类型", "支持WiFi、Bluetooth、UWB等多种无线信号"),
            ("精度评估", "提供完整的定位精度评估和误差分析工具")
        ]

        # 使用两列布局显示功能
        for i, (feature_name, feature_desc) in enumerate(features):
            if i % 2 == 0:
                row_frame = ttk.Frame(features_frame)
                row_frame.pack(fill=tk.X, pady=3)

            feature_item = ttk.Frame(row_frame)
            feature_item.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10) if i % 2 == 0 else (0, 0))

            # 使用项目符号
            bullet = ttk.Label(feature_item, text="●", foreground="#2e5090", font=("Arial", 10))
            bullet.pack(side=tk.LEFT, padx=(0, 5))

            text_frame = ttk.Frame(feature_item)
            text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

            name_label = ttk.Label(text_frame, text=feature_name, font=("Arial", 9, "bold"))
            name_label.pack(anchor=tk.W)

            desc_label = ttk.Label(text_frame, text=feature_desc, font=("Arial", 8), foreground="#666666")
            desc_label.pack(anchor=tk.W)

        # 快捷操作区域
        actions_frame = ttk.LabelFrame(scrollable_frame, text="快捷操作", padding=15)
        actions_frame.pack(fill=tk.X, padx=20, pady=10)

        btn_frame = ttk.Frame(actions_frame)
        btn_frame.pack(fill=tk.X)

        # 创建按钮
        ttk.Button(
            btn_frame,
            text="1. 构建指纹库",
            command=lambda: self.notebook.select(1),
            width=18
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="2. 定位测试",
            command=lambda: self.notebook.select(2),
            width=18
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="3. 非合作定位",
            command=lambda: self.notebook.select(3),
            width=18
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="4. 系统配置",
            command=lambda: self.notebook.select(4),
            width=18
        ).pack(side=tk.LEFT, padx=5)

        # 系统信息区域
        info_frame = ttk.LabelFrame(scrollable_frame, text="系统信息", padding=15)
        info_frame.pack(fill=tk.X, padx=20, pady=(10, 20))

        info_text = scrolledtext.ScrolledText(info_frame, height=6, wrap=tk.WORD, font=("Consolas", 9))
        info_text.pack(fill=tk.BOTH, expand=True)

        info_content = f"""系统版本: v1.0.0
Python版本: {sys.version.split()[0]}
工作目录: {os.getcwd()}

技术支持: 基于射线追踪的电磁仿真和指纹定位技术
适用场景: 室内环境的非合作目标定位、设备追踪、位置感知服务等
"""
        info_text.insert(tk.END, info_content)
        info_text.config(state=tk.DISABLED)

        # 打包canvas和scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 绑定鼠标滚轮
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # 启动状态更新
        self._update_home_status()

    def _update_home_status(self):
        """更新主页状态显示"""
        # 更新模型状态
        if self.model is not None:
            self.home_model_status.config(text="已加载", foreground="green")
        else:
            self.home_model_status.config(text="未加载", foreground="orange")

        # 更新指纹库状态
        if self.fingerprint_db is not None:
            self.home_fp_status.config(text="已加载", foreground="green")
        else:
            self.home_fp_status.config(text="未加载", foreground="orange")

        # 更新定位引擎状态
        if self.localization_engine is not None:
            self.home_engine_status.config(text="已初始化", foreground="green")
        else:
            self.home_engine_status.config(text="未初始化", foreground="orange")

        # 更新跟踪系统状态
        if self.device_tracker is not None:
            if self.tracking_active:
                self.home_tracking_status.config(text="运行中", foreground="blue")
            else:
                self.home_tracking_status.config(text="已初始化", foreground="green")
        else:
            self.home_tracking_status.config(text="未初始化", foreground="orange")

        # 定期更新（每2秒）
        self.root.after(2000, self._update_home_status)

    def _create_build_tab(self):
        """创建构建指纹库选项卡"""

        # 创建Canvas和滚动条来支持页面滚动
        build_canvas = tk.Canvas(self.build_frame)
        build_scrollbar = ttk.Scrollbar(self.build_frame, orient="vertical", command=build_canvas.yview)
        scrollable_build_frame = ttk.Frame(build_canvas)

        scrollable_build_frame.bind(
            "<Configure>",
            lambda e: build_canvas.configure(scrollregion=build_canvas.bbox("all"))
        )

        build_canvas.create_window((0, 0), window=scrollable_build_frame, anchor="nw")
        build_canvas.configure(yscrollcommand=build_scrollbar.set)

        build_canvas.pack(side="left", fill="both", expand=True)
        build_scrollbar.pack(side="right", fill="y")

        # 绑定鼠标滚轮事件
        def _on_mousewheel(event):
            build_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        build_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # 模型加载区域
        model_frame = ttk.LabelFrame(scrollable_build_frame, text="模型加载", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(model_frame, text="3D模型文件:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_path_var = tk.StringVar()
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=50)
        model_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(model_frame, text="浏览...", command=self.browse_model).grid(row=0, column=2, padx=5)

        ttk.Label(model_frame, text="说明: COLLADA (.dae) 文件自动检测单位", foreground="gray").grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Button(model_frame, text="加载模型", command=self.load_model_action, width=15).grid(row=2, column=1, pady=10)

        # 指纹库参数区域
        param_frame = ttk.LabelFrame(scrollable_build_frame, text="指纹库参数", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=10)

        # 定位模式选择
        ttk.Label(param_frame, text="定位模式:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.mode_var = tk.StringVar(value="2D")
        mode_frame = ttk.Frame(param_frame)
        mode_frame.grid(row=0, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Radiobutton(mode_frame, text="2D定位", variable=self.mode_var, value="2D",
                       command=self._toggle_3d_params).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="3D定位", variable=self.mode_var, value="3D",
                       command=self._toggle_3d_params).pack(side=tk.LEFT, padx=5)

        # XY平面网格间距
        ttk.Label(param_frame, text="网格间距 XY (米):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.grid_spacing_var = tk.StringVar(value="1.0")
        ttk.Entry(param_frame, textvariable=self.grid_spacing_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)

        # 2D模式：固定高度
        ttk.Label(param_frame, text="采样高度 (米):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.height_var = tk.StringVar(value="1.5")
        self.height_entry = ttk.Entry(param_frame, textvariable=self.height_var, width=10)
        self.height_entry.grid(row=2, column=1, sticky=tk.W, padx=5)
        self.height_label = ttk.Label(param_frame, text="(仅2D模式)", foreground="gray")
        self.height_label.grid(row=2, column=2, sticky=tk.W, padx=5)

        # 3D模式：Z方向参数
        ttk.Label(param_frame, text="Z轴最小值 (米):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.z_min_var = tk.StringVar(value="0.0")
        self.z_min_entry = ttk.Entry(param_frame, textvariable=self.z_min_var, width=10, state=tk.DISABLED)
        self.z_min_entry.grid(row=3, column=1, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="Z轴最大值 (米):").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.z_max_var = tk.StringVar(value="3.0")
        self.z_max_entry = ttk.Entry(param_frame, textvariable=self.z_max_var, width=10, state=tk.DISABLED)
        self.z_max_entry.grid(row=4, column=1, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="Z轴间距 (米):").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.z_spacing_var = tk.StringVar(value="0.5")
        self.z_spacing_entry = ttk.Entry(param_frame, textvariable=self.z_spacing_var, width=10, state=tk.DISABLED)
        self.z_spacing_entry.grid(row=5, column=1, sticky=tk.W, padx=5)
        self.z_3d_label = ttk.Label(param_frame, text="(仅3D模式)", foreground="gray")
        self.z_3d_label.grid(row=5, column=2, sticky=tk.W, padx=5)

        # AP配置
        ttk.Label(param_frame, text="AP数量:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.num_aps_var = tk.StringVar(value=str(len(FINGERPRINT_CONFIG['ap_positions'])))
        ttk.Label(param_frame, textvariable=self.num_aps_var).grid(row=6, column=1, sticky=tk.W, padx=5)
        ttk.Button(param_frame, text="配置AP位置", command=self.config_aps).grid(row=6, column=2, padx=5)

        # 高精度射线追踪模式
        ttk.Separator(param_frame, orient='horizontal').grid(row=7, column=0, columnspan=3, sticky=tk.EW, pady=10)

        ttk.Label(param_frame, text="射线追踪模式:").grid(row=8, column=0, sticky=tk.W, pady=5)
        self.high_precision_var = tk.BooleanVar(value=False)
        hp_frame = ttk.Frame(param_frame)
        hp_frame.grid(row=8, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Checkbutton(
            hp_frame,
            text="启用高精度反射模式",
            variable=self.high_precision_var,
            command=self._toggle_high_precision
        ).pack(side=tk.LEFT)

        ttk.Label(param_frame, text="最大反射次数:").grid(row=9, column=0, sticky=tk.W, pady=5)
        self.max_reflections_var = tk.StringVar(value="3")
        self.max_reflections_entry = ttk.Entry(param_frame, textvariable=self.max_reflections_var, width=10, state=tk.DISABLED)
        self.max_reflections_entry.grid(row=9, column=1, sticky=tk.W, padx=5)
        self.max_reflections_label = ttk.Label(param_frame, text="(高精度模式)", foreground="gray")
        self.max_reflections_label.grid(row=9, column=2, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="默认材料:").grid(row=10, column=0, sticky=tk.W, pady=5)
        self.default_material_var = tk.StringVar(value="concrete")
        self.default_material_combo = ttk.Combobox(
            param_frame,
            textvariable=self.default_material_var,
            width=15,
            state='disabled',
            values=['concrete', 'brick', 'wood', 'glass', 'metal', 'drywall']
        )
        self.default_material_combo.grid(row=10, column=1, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="场景预设:").grid(row=11, column=0, sticky=tk.W, pady=5)
        self.preset_scene_var = tk.StringVar(value="无")
        self.preset_scene_combo = ttk.Combobox(
            param_frame,
            textvariable=self.preset_scene_var,
            width=15,
            state='disabled',
            values=['无', '办公室', '地下室', '仓库', '住宅']
        )
        self.preset_scene_combo.grid(row=11, column=1, sticky=tk.W, padx=5)
        self.preset_scene_combo.bind('<<ComboboxSelected>>', self._on_preset_selected)

        ttk.Label(param_frame, text="提示: 高精度模式更精确但更慢", foreground="blue").grid(row=12, column=1, columnspan=2, sticky=tk.W, pady=5)

        # 多径传播模式
        ttk.Separator(param_frame, orient='horizontal').grid(row=13, column=0, columnspan=3, sticky=tk.EW, pady=10)

        ttk.Label(param_frame, text="多径传播追踪:").grid(row=14, column=0, sticky=tk.W, pady=5)
        self.multipath_var = tk.BooleanVar(value=False)
        mp_frame = ttk.Frame(param_frame)
        mp_frame.grid(row=14, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Checkbutton(
            mp_frame,
            text="启用多径传播模式",
            variable=self.multipath_var,
            command=self._toggle_multipath
        ).pack(side=tk.LEFT)

        ttk.Label(param_frame, text="射线数量:").grid(row=15, column=0, sticky=tk.W, pady=5)
        self.num_rays_var = tk.StringVar(value="360")
        self.num_rays_entry = ttk.Entry(param_frame, textvariable=self.num_rays_var, width=10, state=tk.DISABLED)
        self.num_rays_entry.grid(row=15, column=1, sticky=tk.W, padx=5)
        self.num_rays_label = ttk.Label(param_frame, text="(多径模式)", foreground="gray")
        self.num_rays_label.grid(row=15, column=2, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="接收容差 (米):").grid(row=16, column=0, sticky=tk.W, pady=5)
        self.rx_tolerance_var = tk.StringVar(value="0.3")
        self.rx_tolerance_entry = ttk.Entry(param_frame, textvariable=self.rx_tolerance_var, width=10, state=tk.DISABLED)
        self.rx_tolerance_entry.grid(row=16, column=1, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="功率阈值 (dBm):").grid(row=17, column=0, sticky=tk.W, pady=5)
        self.power_threshold_var = tk.StringVar(value="-100.0")
        self.power_threshold_entry = ttk.Entry(param_frame, textvariable=self.power_threshold_var, width=10, state=tk.DISABLED)
        self.power_threshold_entry.grid(row=17, column=1, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="提示: 多径模式需同时启用高精度模式，计算最慢但精度最高", foreground="blue").grid(row=18, column=1, columnspan=2, sticky=tk.W, pady=5)

        # 构建按钮
        build_btn_frame = ttk.Frame(scrollable_build_frame)
        build_btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self.build_btn = ttk.Button(
            build_btn_frame,
            text="开始构建指纹库",
            command=self.build_fingerprint_action,
            width=20
        )
        self.build_btn.pack(side=tk.LEFT, padx=5)

        self.visualize_build_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            build_btn_frame,
            text="构建后可视化",
            variable=self.visualize_build_var
        ).pack(side=tk.LEFT, padx=5)

        # 进度条
        self.build_progress = ttk.Progressbar(scrollable_build_frame, mode='determinate', maximum=100)
        self.build_progress.pack(fill=tk.X, padx=10, pady=5)

        # 进度标签
        self.build_progress_label = ttk.Label(scrollable_build_frame, text="", foreground="gray")
        self.build_progress_label.pack(padx=10, pady=2)

    def _create_locate_tab(self):
        """创建定位测试选项卡"""

        # 指纹库加载区域
        fp_frame = ttk.LabelFrame(self.locate_frame, text="指纹库加载", padding=10)
        fp_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(fp_frame, text="指纹库文件:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.fp_path_var = tk.StringVar()
        fp_entry = ttk.Entry(fp_frame, textvariable=self.fp_path_var, width=50)
        fp_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(fp_frame, text="浏览...", command=self.browse_fingerprint).grid(row=0, column=2, padx=5)
        ttk.Button(fp_frame, text="加载指纹库", command=self.load_fingerprint_action, width=15).grid(row=1, column=1, pady=10)

        # 定位算法区域
        algo_frame = ttk.LabelFrame(self.locate_frame, text="定位算法", padding=10)
        algo_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(algo_frame, text="算法:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.algo_var = tk.StringVar(value="wknn")
        algo_combo = ttk.Combobox(
            algo_frame,
            textvariable=self.algo_var,
            values=["knn", "wknn", "probabilistic"],
            width=15,
            state="readonly"
        )
        algo_combo.grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(algo_frame, text="K值:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.k_var = tk.StringVar(value="4")
        ttk.Entry(algo_frame, textvariable=self.k_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)

        # 测试位置区域
        test_frame = ttk.LabelFrame(self.locate_frame, text="测试位置", padding=10)
        test_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(test_frame, text="X (米):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.test_x_var = tk.StringVar(value="5.0")
        ttk.Entry(test_frame, textvariable=self.test_x_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(test_frame, text="Y (米):").grid(row=0, column=2, sticky=tk.W, pady=5, padx=(20, 0))
        self.test_y_var = tk.StringVar(value="5.0")
        ttk.Entry(test_frame, textvariable=self.test_y_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5)

        ttk.Label(test_frame, text="Z (米):").grid(row=0, column=4, sticky=tk.W, pady=5, padx=(20, 0))
        self.test_z_var = tk.StringVar(value="1.5")
        ttk.Entry(test_frame, textvariable=self.test_z_var, width=10).grid(row=0, column=5, sticky=tk.W, padx=5)

        # 定位按钮
        locate_btn_frame = ttk.Frame(self.locate_frame)
        locate_btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            locate_btn_frame,
            text="单点定位测试",
            command=self.single_locate_action,
            width=20
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            locate_btn_frame,
            text="批量定位评估",
            command=self.batch_locate_action,
            width=20
        ).pack(side=tk.LEFT, padx=5)

        self.visualize_locate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            locate_btn_frame,
            text="可视化结果",
            variable=self.visualize_locate_var
        ).pack(side=tk.LEFT, padx=5)

        # 结果显示区域
        result_frame = ttk.LabelFrame(self.locate_frame, text="定位结果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.result_text = scrolledtext.ScrolledText(
            result_frame,
            height=10,
            wrap=tk.WORD,
            font=("Consolas", 10)
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def _create_realtime_tab(self):
        """创建非合作定位选项卡"""

        # 创建Canvas和Scrollbar实现滚动
        self.realtime_canvas = tk.Canvas(self.realtime_frame, highlightthickness=0)
        self.realtime_scrollbar = ttk.Scrollbar(self.realtime_frame, orient="vertical", command=self.realtime_canvas.yview)
        self.realtime_scrollable_frame = ttk.Frame(self.realtime_canvas)

        # 绑定滚动事件
        self.realtime_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.realtime_canvas.configure(scrollregion=self.realtime_canvas.bbox("all"))
        )

        # 创建canvas窗口
        self.realtime_canvas.create_window((0, 0), window=self.realtime_scrollable_frame, anchor="nw")
        self.realtime_canvas.configure(yscrollcommand=self.realtime_scrollbar.set)

        # 打包canvas和scrollbar
        self.realtime_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.realtime_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 绑定鼠标滚轮事件
        def _on_mousewheel(event):
            self.realtime_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.realtime_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # 模式选择区域（现在添加到scrollable_frame）
        mode_frame = ttk.LabelFrame(self.realtime_scrollable_frame, text="工作模式", padding=10)
        mode_frame.pack(fill=tk.X, padx=10, pady=10)

        self.rt_mode_var = tk.StringVar(value="simulated")
        ttk.Radiobutton(
            mode_frame,
            text="模拟模式 (用于测试，无需真实AP)",
            variable=self.rt_mode_var,
            value="simulated",
            command=self._toggle_rt_mode
        ).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(
            mode_frame,
            text="真实AP模式 (需要连接真实AP设备)",
            variable=self.rt_mode_var,
            value="real",
            command=self._toggle_rt_mode
        ).pack(anchor=tk.W, padx=5, pady=2)

        # 真实AP配置（默认禁用）
        ap_config_frame = ttk.LabelFrame(self.realtime_scrollable_frame, text="真实AP配置", padding=10)
        ap_config_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(ap_config_frame, text="AP IP地址 (逗号分隔):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.ap_addresses_var = tk.StringVar(value="192.168.1.101,192.168.1.102,192.168.1.103,192.168.1.104")
        self.ap_addresses_entry = ttk.Entry(ap_config_frame, textvariable=self.ap_addresses_var, width=50, state=tk.DISABLED)
        self.ap_addresses_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(ap_config_frame, text="通信端口:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.ap_port_var = tk.StringVar(value="9999")
        self.ap_port_entry = ttk.Entry(ap_config_frame, textvariable=self.ap_port_var, width=10, state=tk.DISABLED)
        self.ap_port_entry.grid(row=1, column=1, sticky=tk.W, padx=5)

        # 初始化按钮
        init_frame = ttk.Frame(self.realtime_scrollable_frame)
        init_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            init_frame,
            text="初始化跟踪系统",
            command=self.init_tracking_system_action,
            width=20
        ).pack(side=tk.LEFT, padx=5)

        self.rt_status_label = ttk.Label(init_frame, text="状态: 未初始化", foreground="gray")
        self.rt_status_label.pack(side=tk.LEFT, padx=10)

        # 设备管理区域
        device_frame = ttk.LabelFrame(self.realtime_scrollable_frame, text="设备管理", padding=10)
        device_frame.pack(fill=tk.X, padx=10, pady=10)

        # 添加模拟设备（仅模拟模式）
        self.add_device_frame = ttk.Frame(device_frame)
        self.add_device_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.add_device_frame, text="设备标识:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.device_mac_var = tk.StringVar(value="AA:BB:CC:DD:EE:FF")
        ttk.Entry(self.add_device_frame, textvariable=self.device_mac_var, width=20).grid(row=0, column=1, padx=5)

        ttk.Label(self.add_device_frame, text="信号类型:").grid(row=0, column=2, sticky=tk.W, pady=2, padx=(15,0))
        self.signal_type_var = tk.StringVar(value="WiFi")
        signal_type_combo = ttk.Combobox(
            self.add_device_frame,
            textvariable=self.signal_type_var,
            values=["WiFi", "Bluetooth", "Cellular", "RFID", "ZigBee", "LoRa", "UWB"],
            width=12,
            state="readonly"
        )
        signal_type_combo.grid(row=0, column=3, padx=5)
        signal_type_combo.bind("<<ComboboxSelected>>", self._on_signal_type_change)

        ttk.Label(self.add_device_frame, text="位置 (x,y,z):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.device_pos_var = tk.StringVar(value="5,5,1.5")
        ttk.Entry(self.add_device_frame, textvariable=self.device_pos_var, width=15).grid(row=1, column=1, padx=5)

        ttk.Label(self.add_device_frame, text="频率 (Hz):").grid(row=1, column=2, sticky=tk.W, pady=2, padx=(15,0))
        self.frequency_var = tk.StringVar(value="2.4e9")
        ttk.Entry(self.add_device_frame, textvariable=self.frequency_var, width=12).grid(row=1, column=3, padx=5)

        ttk.Label(self.add_device_frame, text="发射功率 (dBm):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.tx_power_device_var = tk.StringVar(value="20.0")
        ttk.Entry(self.add_device_frame, textvariable=self.tx_power_device_var, width=15).grid(row=2, column=1, padx=5)

        ttk.Button(
            self.add_device_frame,
            text="添加模拟设备",
            command=self.add_simulated_device_action,
            width=15
        ).grid(row=2, column=3, padx=5, pady=5)

        # 已添加设备列表
        added_devices_frame = ttk.LabelFrame(device_frame, text="已添加的模拟设备", padding=10)
        added_devices_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 创建已添加设备表格
        added_columns = ("设备标识", "信号类型", "真实位置X", "真实位置Y", "真实位置Z", "频率(GHz)", "功率(dBm)")
        self.added_device_tree = ttk.Treeview(added_devices_frame, columns=added_columns, show="headings", height=5)

        for col in added_columns:
            self.added_device_tree.heading(col, text=col)
            if col == "设备标识":
                self.added_device_tree.column(col, width=130)
            elif col == "信号类型":
                self.added_device_tree.column(col, width=80)
            elif col in ["真实位置X", "真实位置Y", "真实位置Z"]:
                self.added_device_tree.column(col, width=70)
            elif col == "频率(GHz)":
                self.added_device_tree.column(col, width=80)
            elif col == "功率(dBm)":
                self.added_device_tree.column(col, width=80)

        self.added_device_tree.pack(fill=tk.BOTH, expand=True)

        # 添加删除设备按钮
        delete_btn_frame = ttk.Frame(added_devices_frame)
        delete_btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            delete_btn_frame,
            text="删除选中设备",
            command=self.remove_added_device_action,
            width=15
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            delete_btn_frame,
            text="刷新列表",
            command=self.refresh_added_device_list,
            width=15
        ).pack(side=tk.LEFT, padx=5)

        # 控制按钮
        control_frame = ttk.Frame(self.realtime_scrollable_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.start_tracking_btn = ttk.Button(
            control_frame,
            text="开始跟踪",
            command=self.start_tracking_action,
            width=15,
            state=tk.DISABLED
        )
        self.start_tracking_btn.pack(side=tk.LEFT, padx=5)

        self.stop_tracking_btn = ttk.Button(
            control_frame,
            text="停止跟踪",
            command=self.stop_tracking_action,
            width=15,
            state=tk.DISABLED
        )
        self.stop_tracking_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame,
            text="扫描设备",
            command=self.scan_devices_action,
            width=15
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame,
            text="查看实时地图",
            command=self.view_realtime_map_action,
            width=15
        ).pack(side=tk.LEFT, padx=5)

        # 设备列表显示
        list_frame = ttk.LabelFrame(self.realtime_scrollable_frame, text="检测到的设备", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建表格
        columns = ("设备标识", "信号类型", "位置X", "位置Y", "位置Z", "置信度", "最后更新")
        self.device_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)

        for col in columns:
            self.device_tree.heading(col, text=col)
            if col == "设备标识":
                self.device_tree.column(col, width=130)
            elif col == "信号类型":
                self.device_tree.column(col, width=80)
            elif col in ["位置X", "位置Y", "位置Z"]:
                self.device_tree.column(col, width=65)
            elif col == "置信度":
                self.device_tree.column(col, width=65)
            else:
                self.device_tree.column(col, width=120)

        self.device_tree.pack(fill=tk.BOTH, expand=True)

        # 添加刷新按钮
        ttk.Button(list_frame, text="刷新列表", command=self.refresh_device_list).pack(pady=5)

    def _create_config_tab(self):
        """创建系统配置选项卡"""

        # 电磁仿真参数
        em_frame = ttk.LabelFrame(self.config_frame, text="电磁仿真参数", padding=10)
        em_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(em_frame, text="发射功率 (dBm):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.tx_power_var = tk.StringVar(value=str(EM_SIMULATION_CONFIG['tx_power']))
        ttk.Entry(em_frame, textvariable=self.tx_power_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(em_frame, text="工作频率 (GHz):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.freq_var = tk.StringVar(value=str(EM_SIMULATION_CONFIG['tx_frequency'] / 1e9))
        ttk.Entry(em_frame, textvariable=self.freq_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(em_frame, text="最大反射次数:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.max_ref_var = tk.StringVar(value=str(EM_SIMULATION_CONFIG['max_reflections']))
        ttk.Entry(em_frame, textvariable=self.max_ref_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5)

        ttk.Button(em_frame, text="应用配置", command=self.apply_em_config).grid(row=3, column=1, pady=10)

        # 文件路径配置
        path_frame = ttk.LabelFrame(self.config_frame, text="文件路径", padding=10)
        path_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(path_frame, text="模型目录:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Label(path_frame, text=PATHS['models']).grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(path_frame, text="指纹库目录:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(path_frame, text=PATHS['fingerprints']).grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(path_frame, text="结果目录:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Label(path_frame, text=PATHS['results']).grid(row=2, column=1, sticky=tk.W, padx=5)

        # 系统信息
        info_frame = ttk.LabelFrame(self.config_frame, text="系统信息", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        info_text = scrolledtext.ScrolledText(info_frame, height=10, wrap=tk.WORD)
        info_text.pack(fill=tk.BOTH, expand=True)

        info_text.insert(tk.END, "基于几何电磁孪生的室内非合作目标定位系统\n\n")
        info_text.insert(tk.END, "功能特性:\n")
        info_text.insert(tk.END, "  - 支持SketchUp模型导入 (.dae, .obj, .stl)\n")
        info_text.insert(tk.END, "  - 射线追踪电磁仿真\n")
        info_text.insert(tk.END, "  - 无线信号指纹库构建\n")
        info_text.insert(tk.END, "  - K-NN/WKNN/概率定位算法\n")
        info_text.insert(tk.END, "  - 定位精度评估\n")
        info_text.insert(tk.END, "  - 可视化分析\n\n")
        info_text.insert(tk.END, "使用流程:\n")
        info_text.insert(tk.END, "  1. 在'构建指纹库'页面加载模型并构建指纹库\n")
        info_text.insert(tk.END, "  2. 在'定位测试'页面加载指纹库并执行定位\n")
        info_text.insert(tk.END, "  3. 在'系统配置'页面调整参数\n")
        info_text.config(state=tk.DISABLED)

    # ========== 回调函数 ==========

    def _toggle_3d_params(self):
        """切换2D/3D模式参数"""
        mode = self.mode_var.get()
        if mode == "2D":
            # 启用2D参数，禁用3D参数
            self.height_entry.config(state=tk.NORMAL)
            self.z_min_entry.config(state=tk.DISABLED)
            self.z_max_entry.config(state=tk.DISABLED)
            self.z_spacing_entry.config(state=tk.DISABLED)
        else:  # 3D
            # 禁用2D参数，启用3D参数
            self.height_entry.config(state=tk.DISABLED)
            self.z_min_entry.config(state=tk.NORMAL)
            self.z_max_entry.config(state=tk.NORMAL)
            self.z_spacing_entry.config(state=tk.NORMAL)

    def _toggle_high_precision(self):
        """切换高精度模式参数"""
        enabled = self.high_precision_var.get()
        state = tk.NORMAL if enabled else tk.DISABLED
        combobox_state = 'readonly' if enabled else 'disabled'

        self.max_reflections_entry.config(state=state)
        self.default_material_combo.config(state=combobox_state)
        self.preset_scene_combo.config(state=combobox_state)

    def _toggle_multipath(self):
        """切换多径传播模式参数"""
        enabled = self.multipath_var.get()
        state = tk.NORMAL if enabled else tk.DISABLED

        self.num_rays_entry.config(state=state)
        self.rx_tolerance_entry.config(state=state)
        self.power_threshold_entry.config(state=state)

        # 多径模式需要高精度模式
        if enabled and not self.high_precision_var.get():
            messagebox.showwarning("警告", "多径传播模式需要同时启用高精度反射模式")
            self.multipath_var.set(False)
            self.num_rays_entry.config(state=tk.DISABLED)
            self.rx_tolerance_entry.config(state=tk.DISABLED)
            self.power_threshold_entry.config(state=tk.DISABLED)

    def _on_preset_selected(self, event):
        """场景预设选择回调"""
        from material_config_example import (
            OFFICE_CONFIG, BASEMENT_CONFIG,
            WAREHOUSE_CONFIG, RESIDENTIAL_CONFIG
        )

        preset = self.preset_scene_var.get()
        if preset == '办公室':
            self.max_reflections_var.set("2")
            self.default_material_var.set("drywall")
            self._current_preset_config = OFFICE_CONFIG
            self.log("已加载办公室场景预设配置")
        elif preset == '地下室':
            self.max_reflections_var.set("4")
            self.default_material_var.set("concrete")
            self._current_preset_config = BASEMENT_CONFIG
            self.log("已加载地下室场景预设配置")
        elif preset == '仓库':
            self.max_reflections_var.set("3")
            self.default_material_var.set("concrete")
            self._current_preset_config = WAREHOUSE_CONFIG
            self.log("已加载仓库场景预设配置")
        elif preset == '住宅':
            self.max_reflections_var.set("2")
            self.default_material_var.set("brick")
            self._current_preset_config = RESIDENTIAL_CONFIG
            self.log("已加载住宅场景预设配置")
        else:
            self._current_preset_config = None

    def browse_model(self):
        """浏览模型文件"""
        filename = filedialog.askopenfilename(
            title="选择3D模型文件",
            filetypes=[
                ("COLLADA files", "*.dae"),
                ("Wavefront files", "*.obj"),
                ("STL files", "*.stl"),
                ("All files", "*.*")
            ],
            initialdir=PATHS['models']
        )
        if filename:
            self.model_path_var.set(filename)

    def browse_fingerprint(self):
        """浏览指纹库文件"""
        filename = filedialog.askopenfilename(
            title="选择指纹库文件",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=PATHS['fingerprints']
        )
        if filename:
            self.fp_path_var.set(filename)

    def load_model_action(self):
        """加载模型"""
        model_path = self.model_path_var.get()
        if not model_path:
            messagebox.showerror("错误", "请先选择模型文件")
            return

        try:
            self.log("正在加载模型...")
            # 所有文件都使用自动单位检测
            file_ext = os.path.splitext(model_path)[1].lower()
            if file_ext == '.dae':
                self.log("COLLADA 文件：启用自动单位检测")
            else:
                self.log(f"{file_ext.upper()} 文件：使用默认单位设置")

            self.model = load_model(model_path, unit='auto')
            self.model_path = model_path

            # 创建射线追踪器
            self.log("正在初始化电磁仿真引擎...")
            self.ray_tracer = create_ray_tracer(self.model, EM_SIMULATION_CONFIG)

            self.log("模型加载成功！")
            messagebox.showinfo("成功", "模型加载成功！")

        except Exception as e:
            self.log(f"错误: {str(e)}")
            messagebox.showerror("错误", f"加载模型失败:\n{str(e)}")

    def config_aps(self):
        """配置AP位置"""
        # 创建AP配置对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("配置AP位置")
        dialog.geometry("400x300")

        ttk.Label(dialog, text="请输入AP位置 (格式: x,y,z 每行一个)", font=("Arial", 10)).pack(pady=10)

        text = scrolledtext.ScrolledText(dialog, height=10)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 预填充当前AP位置
        for ap_pos in FINGERPRINT_CONFIG['ap_positions']:
            text.insert(tk.END, f"{ap_pos[0]},{ap_pos[1]},{ap_pos[2]}\n")

        def save_aps():
            try:
                lines = text.get("1.0", tk.END).strip().split("\n")
                new_aps = []
                for line in lines:
                    if line.strip():
                        x, y, z = map(float, line.split(","))
                        new_aps.append((x, y, z))

                FINGERPRINT_CONFIG['ap_positions'] = new_aps
                self.num_aps_var.set(str(len(new_aps)))
                self.log(f"已更新AP位置，共 {len(new_aps)} 个AP")
                messagebox.showinfo("成功", f"已保存 {len(new_aps)} 个AP位置")
                dialog.destroy()

            except Exception as e:
                messagebox.showerror("错误", f"解析AP位置失败:\n{str(e)}")

        ttk.Button(dialog, text="保存", command=save_aps).pack(pady=10)

    def build_fingerprint_action(self):
        """构建指纹库"""
        if self.model is None or self.ray_tracer is None:
            messagebox.showerror("错误", "请先加载模型")
            return

        mode = self.mode_var.get()

        try:
            grid_spacing = float(self.grid_spacing_var.get())

            if mode == "2D":
                height = float(self.height_var.get())
                z_min = None
                z_max = None
                z_spacing = None
            else:  # 3D
                height = None
                z_min = float(self.z_min_var.get())
                z_max = float(self.z_max_var.get())
                z_spacing = float(self.z_spacing_var.get())

            # 高精度模式参数
            high_precision = self.high_precision_var.get()
            if high_precision:
                max_reflections = int(self.max_reflections_var.get())
                default_material = self.default_material_var.get()
            else:
                max_reflections = None
                default_material = None

            # 多径传播模式参数
            multipath_enabled = self.multipath_var.get()
            if multipath_enabled:
                num_rays = int(self.num_rays_var.get())
                rx_tolerance = float(self.rx_tolerance_var.get())
                power_threshold_dbm = float(self.power_threshold_var.get())
            else:
                num_rays = None
                rx_tolerance = None
                power_threshold_dbm = None

        except ValueError:
            messagebox.showerror("错误", "参数格式错误")
            return

        # 进度回调函数
        def update_progress(current, total, percent):
            self.build_progress['value'] = percent
            self.build_progress_label.config(text=f"进度: {current}/{total} ({percent:.1f}%)")
            self.root.update_idletasks()

        # 在新线程中执行
        def build_task():
            try:
                self.build_btn.config(state=tk.DISABLED)
                self.build_progress['value'] = 0
                self.build_progress_label.config(text="准备开始...")

                self.log(f"开始构建指纹库 ({mode}模式)...")
                self.log("步骤1: 准备配置...")
                config = FINGERPRINT_CONFIG.copy()
                config['grid_spacing'] = grid_spacing
                config['height'] = height
                config['z_min'] = z_min
                config['z_max'] = z_max
                config['z_spacing'] = z_spacing

                # 高精度模式或多径模式：重新创建ray_tracer
                ray_tracer_to_use = self.ray_tracer
                if high_precision or multipath_enabled:
                    self.log("步骤1.5: 配置射线追踪模式...")
                    em_config = EM_SIMULATION_CONFIG.copy()

                    if high_precision:
                        em_config['high_precision_mode'] = True
                        em_config['max_reflections'] = max_reflections
                        em_config['default_material'] = default_material

                        # 如果选择了场景预设，使用预设配置
                        if self._current_preset_config:
                            self.log(f"使用场景预设配置: {self.preset_scene_var.get()}")
                            em_config.update(self._current_preset_config)
                        else:
                            # 否则使用基本配置
                            em_config['custom_materials'] = {}

                        self.log(f"高精度模式已启用 (最大反射次数: {max_reflections}, 默认材料: {default_material})")

                    if multipath_enabled:
                        em_config['multipath_enabled'] = True
                        em_config['num_rays'] = num_rays
                        em_config['rx_tolerance'] = rx_tolerance
                        em_config['power_threshold_dbm'] = power_threshold_dbm
                        # 多径模式必须启用高精度
                        em_config['high_precision_mode'] = True
                        if not high_precision:
                            em_config['max_reflections'] = 3
                            em_config['default_material'] = 'concrete'
                        self.log(f"多径传播模式已启用 (射线数: {num_rays}, 接收容差: {rx_tolerance}m, 功率阈值: {power_threshold_dbm}dBm)")

                    # 重新创建ray_tracer
                    from src.simulation import create_ray_tracer
                    ray_tracer_to_use = create_ray_tracer(self.model, em_config)

                self.log("步骤2: 创建指纹库构建器...")
                # 手动构建以支持进度回调（使用批量模式加速）
                from src.fingerprint.builder import FingerprintBuilder
                builder = FingerprintBuilder(self.model, ray_tracer_to_use, config)

                self.log("步骤3: 开始构建指纹库（批量模式）...")
                self.fingerprint_db = builder.build(
                    grid_spacing=grid_spacing,
                    height=height,
                    z_min=z_min,
                    z_max=z_max,
                    z_spacing=z_spacing,
                    progress_callback=update_progress,
                    batch_size=None  # None表示一次处理所有点（最快）
                )

                # 保存
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(PATHS['fingerprints'], f'fingerprint_{timestamp}.pkl')
                self.fingerprint_db.save(save_path)
                self.fingerprint_path = save_path

                self.log(f"指纹库已保存到: {save_path}")

                # 可视化 (使用高性能Plotly)
                if self.visualize_build_var.get():
                    self.log("正在生成可视化...")
                    viz = VisualizerPlotly()
                    viz.plot_all_aps_heatmap(
                        self.fingerprint_db,
                        save_path=os.path.join(PATHS['results'], 'heatmap_all_aps.html')
                    )

                self.log("指纹库构建完成！")
                messagebox.showinfo("成功", "指纹库构建完成！")

            except Exception as e:
                self.log(f"错误: {str(e)}")
                messagebox.showerror("错误", f"构建指纹库失败:\n{str(e)}")

            finally:
                self.build_progress['value'] = 0
                self.build_progress_label.config(text="")
                self.build_btn.config(state=tk.NORMAL)

        thread = threading.Thread(target=build_task)
        thread.daemon = True
        thread.start()

    def load_fingerprint_action(self):
        """加载指纹库"""
        fp_path = self.fp_path_var.get()
        if not fp_path:
            messagebox.showerror("错误", "请先选择指纹库文件")
            return

        try:
            self.log("正在加载指纹库...")
            self.fingerprint_db = FingerprintDatabase.load(fp_path)
            self.fingerprint_path = fp_path

            # 创建定位引擎
            config = LOCALIZATION_CONFIG.copy()
            config['algorithm'] = self.algo_var.get()
            config['k_neighbors'] = int(self.k_var.get())

            self.localization_engine = create_localization_engine(
                self.fingerprint_db, config
            )

            self.log("指纹库加载成功！")
            messagebox.showinfo("成功", "指纹库加载成功！")

        except Exception as e:
            self.log(f"错误: {str(e)}")
            messagebox.showerror("错误", f"加载指纹库失败:\n{str(e)}")

    def single_locate_action(self):
        """单点定位测试"""
        if self.fingerprint_db is None or self.localization_engine is None:
            messagebox.showerror("错误", "请先加载指纹库")
            return

        try:
            test_x = float(self.test_x_var.get())
            test_y = float(self.test_y_var.get())
            test_z = float(self.test_z_var.get())
            test_pos = np.array([test_x, test_y, test_z])

            self.log(f"测试位置: [{test_x:.2f}, {test_y:.2f}, {test_z:.2f}]")

            # 获取RSSI（从指纹库或使用最近点）
            measured_rssi = self.fingerprint_db.get_fingerprint(tuple(test_pos))
            if measured_rssi is None:
                positions, rssi_matrix = self.fingerprint_db.get_all_fingerprints()
                distances = np.linalg.norm(positions - test_pos, axis=1)
                nearest_idx = np.argmin(distances)
                measured_rssi = rssi_matrix[nearest_idx]
                self.log(f"使用最近指纹点 (距离 {distances[nearest_idx]:.2f}m)")

            # 定位
            result = self.localization_engine.locate(measured_rssi)

            # 计算误差
            error = np.linalg.norm(test_pos[:2] - result['position'][:2])

            # 显示结果
            result_str = f"\n定位结果:\n"
            result_str += f"  估计位置: [{result['position'][0]:.2f}, {result['position'][1]:.2f}, {result['position'][2]:.2f}] 米\n"
            result_str += f"  置信度: {result['confidence']:.3f}\n"
            result_str += f"  定位误差: {error:.2f} 米\n"
            result_str += f"  算法: {result['algorithm']}\n"

            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, result_str)
            self.log(result_str)

            # 可视化 (使用高性能Plotly - GPU加速，WebGL渲染)
            if self.visualize_locate_var.get():
                viz = VisualizerPlotly()
                # Plotly具有更高的性能，可以流畅处理大量数据点
                viz.plot_localization_result(
                    test_pos,
                    result['position'],
                    self.fingerprint_db,
                    save_path=os.path.join(PATHS['results'], 'localization_result.html'),
                    show_fingerprints=True,  # Plotly可以轻松处理大量点
                    downsample_factor=5  # 由于性能更好，可以显示更多点
                )

        except Exception as e:
            self.log(f"错误: {str(e)}")
            messagebox.showerror("错误", f"定位失败:\n{str(e)}")

    def batch_locate_action(self):
        """批量定位评估"""
        if self.fingerprint_db is None or self.localization_engine is None:
            messagebox.showerror("错误", "请先加载指纹库")
            return

        def batch_task():
            try:
                self.log("开始批量定位评估...")

                positions, rssi_matrix = self.fingerprint_db.get_all_fingerprints()

                # 使用20%数据作为测试集
                test_ratio = 0.2
                num_test = int(len(positions) * test_ratio)
                test_indices = np.random.choice(len(positions), num_test, replace=False)

                test_positions = positions[test_indices]
                test_rssi = rssi_matrix[test_indices]

                # 评估
                eval_result = self.localization_engine.evaluate_accuracy(
                    test_positions, test_rssi
                )

                # 显示结果
                result_str = f"\n批量定位评估结果 (测试样本: {num_test}):\n"
                result_str += f"  平均误差: {eval_result['mean_error']:.2f} 米\n"
                result_str += f"  中位误差: {eval_result['median_error']:.2f} 米\n"
                result_str += f"  标准差: {eval_result['std_error']:.2f} 米\n"
                result_str += f"  最大误差: {eval_result['max_error']:.2f} 米\n"

                self.result_text.delete("1.0", tk.END)
                self.result_text.insert(tk.END, result_str)
                self.log(result_str)

                # 可视化CDF (使用高性能Plotly)
                if self.visualize_locate_var.get():
                    viz = VisualizerPlotly()
                    viz.plot_error_cdf(
                        eval_result['errors'],
                        save_path=os.path.join(PATHS['results'], 'error_cdf.html')
                    )

                messagebox.showinfo("完成", "批量定位评估完成！")

            except Exception as e:
                self.log(f"错误: {str(e)}")
                messagebox.showerror("错误", f"批量评估失败:\n{str(e)}")

        thread = threading.Thread(target=batch_task)
        thread.daemon = True
        thread.start()

    def apply_em_config(self):
        """应用电磁仿真配置"""
        try:
            EM_SIMULATION_CONFIG['tx_power'] = float(self.tx_power_var.get())
            EM_SIMULATION_CONFIG['tx_frequency'] = float(self.freq_var.get()) * 1e9
            EM_SIMULATION_CONFIG['max_reflections'] = int(self.max_ref_var.get())

            self.log("电磁仿真参数已更新")
            messagebox.showinfo("成功", "配置已应用")

        except Exception as e:
            messagebox.showerror("错误", f"应用配置失败:\n{str(e)}")

    def log(self, message):
        """输出日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        self.log_text.insert(tk.END, log_msg + "\n")
        self.log_text.see(tk.END)
        self.root.update()
        # 同时输出到控制台，方便调试
        print(log_msg)

    def clear_log(self):
        """清空日志"""
        self.log_text.delete("1.0", tk.END)

    def _save_settings(self):
        """保存GUI设置"""
        settings = {
            # 模型设置
            'model_path': self.model_path_var.get(),

            # 指纹库参数
            'mode': self.mode_var.get(),
            'grid_spacing': self.grid_spacing_var.get(),
            'height': self.height_var.get(),
            'z_min': self.z_min_var.get(),
            'z_max': self.z_max_var.get(),
            'z_spacing': self.z_spacing_var.get(),

            # AP配置
            'ap_positions': FINGERPRINT_CONFIG['ap_positions'],

            # 定位参数
            'fp_path': self.fp_path_var.get(),
            'algorithm': self.algo_var.get(),
            'k': self.k_var.get(),
            'test_x': self.test_x_var.get(),
            'test_y': self.test_y_var.get(),
            'test_z': self.test_z_var.get(),

            # 电磁仿真参数
            'tx_power': self.tx_power_var.get(),
            'frequency': self.freq_var.get(),
            'max_reflections': self.max_ref_var.get(),

            # 可视化选项
            'visualize_build': self.visualize_build_var.get(),
            'visualize_locate': self.visualize_locate_var.get()
        }

        try:
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存设置失败: {e}")

    def _load_settings(self):
        """加载GUI设置"""
        if not os.path.exists(SETTINGS_FILE):
            return

        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            # 恢复模型设置
            if 'model_path' in settings:
                self.model_path_var.set(settings['model_path'])

            # 恢复指纹库参数
            if 'mode' in settings:
                self.mode_var.set(settings['mode'])
                self._toggle_3d_params()  # 更新UI状态
            if 'grid_spacing' in settings:
                self.grid_spacing_var.set(settings['grid_spacing'])
            if 'height' in settings:
                self.height_var.set(settings['height'])
            if 'z_min' in settings:
                self.z_min_var.set(settings['z_min'])
            if 'z_max' in settings:
                self.z_max_var.set(settings['z_max'])
            if 'z_spacing' in settings:
                self.z_spacing_var.set(settings['z_spacing'])

            # 恢复AP配置
            if 'ap_positions' in settings:
                FINGERPRINT_CONFIG['ap_positions'] = settings['ap_positions']
                self.num_aps_var.set(str(len(settings['ap_positions'])))

            # 恢复定位参数
            if 'fp_path' in settings:
                self.fp_path_var.set(settings['fp_path'])
            if 'algorithm' in settings:
                self.algo_var.set(settings['algorithm'])
            if 'k' in settings:
                self.k_var.set(settings['k'])
            if 'test_x' in settings:
                self.test_x_var.set(settings['test_x'])
            if 'test_y' in settings:
                self.test_y_var.set(settings['test_y'])
            if 'test_z' in settings:
                self.test_z_var.set(settings['test_z'])

            # 恢复电磁仿真参数
            if 'tx_power' in settings:
                self.tx_power_var.set(settings['tx_power'])
            if 'frequency' in settings:
                self.freq_var.set(settings['frequency'])
            if 'max_reflections' in settings:
                self.max_ref_var.set(settings['max_reflections'])

            # 恢复可视化选项
            if 'visualize_build' in settings:
                self.visualize_build_var.set(settings['visualize_build'])
            if 'visualize_locate' in settings:
                self.visualize_locate_var.set(settings['visualize_locate'])

            self.log("已加载上次保存的设置")

        except Exception as e:
            print(f"加载设置失败: {e}")

    # ========== 非合作定位回调函数 ==========

    def _toggle_rt_mode(self):
        """切换实时定位模式"""
        mode = self.rt_mode_var.get()
        if mode == "simulated":
            # 启用模拟设备添加，禁用AP配置
            for widget in self.add_device_frame.winfo_children():
                if isinstance(widget, (ttk.Entry, ttk.Button, ttk.Combobox)):
                    widget.config(state=tk.NORMAL if not isinstance(widget, ttk.Combobox) else "readonly")
            self.ap_addresses_entry.config(state=tk.DISABLED)
            self.ap_port_entry.config(state=tk.DISABLED)
        else:  # real
            # 禁用模拟设备添加，启用AP配置
            for widget in self.add_device_frame.winfo_children():
                if isinstance(widget, (ttk.Entry, ttk.Button, ttk.Combobox)):
                    widget.config(state=tk.DISABLED)
            self.ap_addresses_entry.config(state=tk.NORMAL)
            self.ap_port_entry.config(state=tk.NORMAL)

    def _on_signal_type_change(self, event=None):
        """信号类型改变时自动更新参数"""
        from config import EM_SIGNAL_TRACKING_CONFIG

        signal_type = self.signal_type_var.get()
        if signal_type in EM_SIGNAL_TRACKING_CONFIG['signal_types']:
            params = EM_SIGNAL_TRACKING_CONFIG['signal_types'][signal_type]
            self.frequency_var.set(str(params['frequency']))
            self.tx_power_device_var.set(str(params['tx_power']))

    def init_tracking_system_action(self):
        """初始化跟踪系统"""
        if self.fingerprint_db is None or self.localization_engine is None:
            messagebox.showerror("错误", "请先在'定位测试'页面加载指纹库")
            return

        # 建议用户加载模型（用于可视化）
        if self.model is None:
            response = messagebox.askyesno(
                "提示",
                "未检测到3D模型\n\n" +
                "加载模型后可在实时地图中显示室内环境\n" +
                "是否现在去'构建指纹库'页面加载模型？\n\n" +
                "（选择'否'将继续初始化，但实时地图不显示模型）"
            )
            if response:  # 用户选择"是"
                self.notebook.select(1)  # 切换到"构建指纹库"选项卡
                messagebox.showinfo("提示", "请加载模型后再返回此页面初始化跟踪系统")
                return
            else:
                self.log("提示: 未加载模型，实时地图将不显示室内环境")

        try:
            mode = self.rt_mode_var.get()
            ap_positions = FINGERPRINT_CONFIG['ap_positions']

            if mode == "simulated":
                # 创建通用电磁信号采集器（模拟模式）
                self.signal_collector = UniversalEMSignalCollector(
                    receiver_positions=ap_positions,
                    fingerprint_db=self.fingerprint_db,
                    ray_tracer=self.ray_tracer
                )
                self.log("已初始化通用电磁信号采集器（模拟模式）")
                self.log("支持信号类型: WiFi, Bluetooth, Cellular, RFID, ZigBee, LoRa, UWB")
            else:
                # 创建真实电磁信号采集器
                ap_addresses_str = self.ap_addresses_var.get()
                ap_addresses = [addr.strip() for addr in ap_addresses_str.split(",")]
                ap_port = int(self.ap_port_var.get())

                if len(ap_addresses) != len(ap_positions):
                    messagebox.showerror("错误", f"接收机数量不匹配：需要{len(ap_positions)}个，提供了{len(ap_addresses)}个")
                    return

                # 构建接收机配置
                receiver_configs = []
                for i, (addr, pos) in enumerate(zip(ap_addresses, ap_positions)):
                    receiver_configs.append({
                        'address': addr,
                        'port': ap_port,
                        'protocol': 'udp',
                        'type': 'Universal'  # 通用接收机
                    })

                self.signal_collector = RealEMSignalCollector(
                    receiver_configs=receiver_configs,
                    fingerprint_db=self.fingerprint_db
                )
                self.log(f"已连接到{len(receiver_configs)}个真实接收机")

            # 创建设备跟踪器
            self.device_tracker = DeviceTracker(
                signal_collector=self.signal_collector,
                localization_engine=self.localization_engine,
                update_interval=1.0
            )

            self.rt_status_label.config(text=f"状态: 已初始化 ({mode}模式)", foreground="green")
            self.start_tracking_btn.config(state=tk.NORMAL)
            self.log("跟踪系统初始化完成")
            messagebox.showinfo("成功", "跟踪系统初始化完成！\n支持追踪多种电磁信号类型")

        except Exception as e:
            self.log(f"错误: {str(e)}")
            messagebox.showerror("错误", f"初始化失败:\n{str(e)}")

    def add_simulated_device_action(self):
        """添加模拟设备"""
        if self.signal_collector is None or not isinstance(self.signal_collector, UniversalEMSignalCollector):
            messagebox.showerror("错误", "请先在模拟模式下初始化跟踪系统")
            return

        try:
            # 获取参数
            mac = self.device_mac_var.get()
            pos_str = self.device_pos_var.get()
            x, y, z = map(float, pos_str.split(","))
            position = np.array([x, y, z])

            signal_type = self.signal_type_var.get()
            frequency = float(self.frequency_var.get())
            tx_power = float(self.tx_power_device_var.get())

            # 获取路径损耗指数
            from config import EM_SIGNAL_TRACKING_CONFIG
            if signal_type in EM_SIGNAL_TRACKING_CONFIG['signal_types']:
                path_loss_exponent = EM_SIGNAL_TRACKING_CONFIG['signal_types'][signal_type]['path_loss_exponent']
            else:
                path_loss_exponent = 2.0  # 默认值

            # 创建电磁目标
            em_target = EMTarget(
                mac=mac,
                position=position,
                signal_type=signal_type,
                frequency=frequency,
                tx_power=tx_power,
                path_loss_exponent=path_loss_exponent
            )

            # 添加到信号采集器
            self.signal_collector.add_target(em_target)

            # 添加到设备跟踪器
            self.device_tracker.add_device(
                mac=mac,
                name=f"{signal_type}设备_{mac[-5:]}",
                signal_type=signal_type,
                frequency=frequency,
                tx_power=tx_power
            )

            self.log(f"已添加{signal_type}设备: {mac} @ [{x:.2f}, {y:.2f}, {z:.2f}]")
            self.log(f"  频率: {frequency/1e9:.2f} GHz, 发射功率: {tx_power} dBm")
            messagebox.showinfo("成功", f"已添加{signal_type}设备\n{mac}")

            # 刷新设备列表
            self.refresh_device_list()
            # 刷新已添加设备列表
            self.refresh_added_device_list()

        except Exception as e:
            self.log(f"添加设备失败: {str(e)}")
            messagebox.showerror("错误", f"添加设备失败:\n{str(e)}")

    def refresh_added_device_list(self):
        """刷新已添加设备列表"""
        if self.signal_collector is None or not isinstance(self.signal_collector, UniversalEMSignalCollector):
            return

        try:
            # 清空列表
            for item in self.added_device_tree.get_children():
                self.added_device_tree.delete(item)

            # 获取所有已添加的EM目标
            for mac, em_target in self.signal_collector.em_targets.items():
                values = (
                    mac,
                    em_target.signal_type,
                    f"{em_target.position[0]:.2f}",
                    f"{em_target.position[1]:.2f}",
                    f"{em_target.position[2]:.2f}",
                    f"{em_target.frequency/1e9:.2f}",
                    f"{em_target.tx_power:.1f}"
                )
                self.added_device_tree.insert("", tk.END, values=values)

        except Exception as e:
            self.log(f"刷新已添加设备列表错误: {str(e)}")

    def remove_added_device_action(self):
        """删除选中的已添加设备"""
        if self.signal_collector is None or not isinstance(self.signal_collector, UniversalEMSignalCollector):
            messagebox.showerror("错误", "请先在模拟模式下初始化跟踪系统")
            return

        # 获取选中的设备
        selected_items = self.added_device_tree.selection()
        if not selected_items:
            messagebox.showwarning("提示", "请先选择要删除的设备")
            return

        try:
            # 确认删除
            response = messagebox.askyesno("确认", f"确定要删除选中的 {len(selected_items)} 个设备吗？")
            if not response:
                return

            # 删除每个选中的设备
            for item in selected_items:
                values = self.added_device_tree.item(item, 'values')
                mac = values[0]  # 第一列是设备标识

                # 从信号采集器删除
                if mac in self.signal_collector.em_targets:
                    del self.signal_collector.em_targets[mac]
                    self.log(f"已从信号采集器删除设备: {mac}")

                # 从设备跟踪器删除
                if self.device_tracker is not None:
                    # DeviceTracker没有直接删除方法，所以只能从信号采集器删除
                    pass

            # 刷新列表
            self.refresh_added_device_list()
            messagebox.showinfo("成功", f"已删除 {len(selected_items)} 个设备")

        except Exception as e:
            self.log(f"删除设备失败: {str(e)}")
            messagebox.showerror("错误", f"删除设备失败:\n{str(e)}")

    def start_tracking_action(self):
        """开始跟踪"""
        if self.device_tracker is None:
            messagebox.showerror("错误", "请先初始化跟踪系统")
            return

        try:
            self.device_tracker.start_tracking(auto_discover=True)
            self.tracking_active = True

            self.start_tracking_btn.config(state=tk.DISABLED)
            self.stop_tracking_btn.config(state=tk.NORMAL)
            self.rt_status_label.config(text="状态: 跟踪中...", foreground="blue")

            self.log("开始实时跟踪...")

            # 启动定时刷新设备列表
            self._refresh_device_list_periodically()

        except Exception as e:
            messagebox.showerror("错误", f"开始跟踪失败:\n{str(e)}")

    def stop_tracking_action(self):
        """停止跟踪"""
        if self.device_tracker is None:
            return

        try:
            self.device_tracker.stop_tracking()
            self.tracking_active = False

            self.start_tracking_btn.config(state=tk.NORMAL)
            self.stop_tracking_btn.config(state=tk.DISABLED)
            self.rt_status_label.config(text="状态: 已停止", foreground="gray")

            self.log("停止实时跟踪")

        except Exception as e:
            messagebox.showerror("错误", f"停止跟踪失败:\n{str(e)}")

    def scan_devices_action(self):
        """扫描设备"""
        if self.signal_collector is None:
            messagebox.showerror("错误", "请先初始化跟踪系统")
            return

        try:
            self.log("正在扫描设备...")
            devices = self.signal_collector.scan_devices()

            if devices:
                self.log(f"发现 {len(devices)} 个设备: {', '.join(devices)}")
                messagebox.showinfo("扫描结果", f"发现 {len(devices)} 个设备:\n" + "\n".join(devices))
            else:
                self.log("未发现任何设备")
                messagebox.showinfo("扫描结果", "未发现任何设备")

            self.refresh_device_list()

        except Exception as e:
            messagebox.showerror("错误", f"扫描失败:\n{str(e)}")

    def refresh_device_list(self):
        """刷新设备列表"""
        if self.device_tracker is None:
            return

        try:
            # 清空列表
            for item in self.device_tree.get_children():
                self.device_tree.delete(item)

            # 获取所有设备
            devices = self.device_tracker.get_all_devices()

            for device in devices:
                if device.position is not None:
                    values = (
                        device.mac,
                        device.signal_type,
                        f"{device.position[0]:.2f}",
                        f"{device.position[1]:.2f}",
                        f"{device.position[2]:.2f}",
                        f"{device.confidence:.3f}",
                        device.last_seen.strftime("%H:%M:%S")
                    )
                    self.device_tree.insert("", tk.END, values=values)

        except Exception as e:
            self.log(f"刷新列表错误: {str(e)}")

    def _refresh_device_list_periodically(self):
        """定期刷新设备列表"""
        if self.tracking_active:
            self.refresh_device_list()
            # 每2秒刷新一次
            self.root.after(2000, self._refresh_device_list_periodically)

    def view_realtime_map_action(self):
        """查看实时地图"""
        if self.device_tracker is None:
            messagebox.showerror("错误", "请先初始化跟踪系统")
            return

        try:
            devices = self.device_tracker.get_active_devices()

            if not devices:
                messagebox.showinfo("提示", "当前没有活跃设备")
                return

            self.log("正在生成实时地图...")

            # 收集设备的真实位置和预测位置
            predicted_positions = []
            predicted_labels = []
            predicted_colors = []

            true_positions = []
            true_labels = []
            true_colors = []

            error_lines = []  # 存储误差连线

            # 为不同信号类型定义颜色
            signal_type_colors = {
                'WiFi': 'blue',
                'Bluetooth': 'cyan',
                'Cellular': 'purple',
                'RFID': 'orange',
                'ZigBee': 'green',
                'LoRa': 'brown',
                'UWB': 'magenta'
            }

            for device in devices:
                # 预测位置（来自定位引擎）
                if device.position is not None:
                    predicted_positions.append(device.position)
                    error = 0.0

                    # 尝试获取真实位置（来自EMTarget）
                    true_pos = None
                    if isinstance(self.signal_collector, UniversalEMSignalCollector):
                        # 从信号采集器获取真实位置
                        if device.mac in self.signal_collector.em_targets:
                            em_target = self.signal_collector.em_targets[device.mac]
                            true_pos = em_target.position
                            true_positions.append(true_pos)
                            true_labels.append(f"{device.signal_type}<br>{device.mac[-8:]}<br>(真实)")
                            true_colors.append('green')

                            # 计算误差
                            error = np.linalg.norm(device.position - true_pos)

                            # 添加误差连线
                            error_lines.append({
                                'predicted': device.position,
                                'true': true_pos
                            })

                    # 预测位置标签（显示误差）
                    if error > 0:
                        predicted_labels.append(f"{device.signal_type}<br>{device.mac[-8:]}<br>(预测, 误差{error:.2f}m)")
                    else:
                        predicted_labels.append(f"{device.signal_type}<br>{device.mac[-8:]}<br>(预测)")

                    predicted_colors.append(signal_type_colors.get(device.signal_type, 'blue'))

            predicted_positions = np.array(predicted_positions) if predicted_positions else np.array([]).reshape(0,3)
            true_positions = np.array(true_positions) if true_positions else np.array([]).reshape(0,3)

            # 使用Plotly可视化（传入模型）
            self.log(f"模型状态: {'已加载' if self.model is not None else '未加载'}")
            if self.model is not None and self.model.mesh is not None:
                self.log(f"模型顶点数: {len(self.model.mesh.vertices)}, 面片数: {len(self.model.mesh.faces)}")

            viz = VisualizerPlotly(model=self.model)

            # 创建3D图形
            import plotly.graph_objects as go

            fig = go.Figure()

            # 添加3D模型
            self.log("正在添加3D模型到图形...")
            viz._add_model_to_figure(fig)
            self.log("3D模型添加完成")

            # 添加指纹点（降采样）
            if self.fingerprint_db is not None:
                positions, _ = self.fingerprint_db.get_all_fingerprints()
                sample_indices = np.random.choice(len(positions), min(500, len(positions)), replace=False)
                sample_positions = positions[sample_indices]

                fig.add_trace(go.Scatter3d(
                    x=sample_positions[:, 0],
                    y=sample_positions[:, 1],
                    z=sample_positions[:, 2],
                    mode='markers',
                    marker=dict(size=2, color='lightgray', opacity=0.2, line=dict(width=0)),
                    name='指纹点',
                    showlegend=True,
                    hoverinfo='skip'
                ))

            # 添加接收机位置（原AP位置）
            ap_positions = np.array(FINGERPRINT_CONFIG['ap_positions'])
            fig.add_trace(go.Scatter3d(
                x=ap_positions[:, 0],
                y=ap_positions[:, 1],
                z=ap_positions[:, 2],
                mode='markers+text',
                marker=dict(size=10, color='red', symbol='diamond', line=dict(width=2, color='black')),
                text=[f'接收机{i+1}' for i in range(len(ap_positions))],
                textposition='top center',
                name='接收机',
                showlegend=True,
                hovertemplate='<b>接收机%{text}</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<extra></extra>',
                customdata=[[f'{i+1}'] for i in range(len(ap_positions))]
            ))

            # 添加真实位置（绿色圆圈）
            if len(true_positions) > 0:
                # 准备自定义悬停数据
                true_hover_text = []
                for i, device in enumerate(devices):
                    if isinstance(self.signal_collector, UniversalEMSignalCollector):
                        if device.mac in self.signal_collector.em_targets:
                            em_target = self.signal_collector.em_targets[device.mac]
                            true_hover_text.append(
                                f"<b>{device.signal_type} - {device.mac[-8:]}</b><br>" +
                                f"<b>真实位置:</b><br>" +
                                f"X: {em_target.position[0]:.2f}m<br>" +
                                f"Y: {em_target.position[1]:.2f}m<br>" +
                                f"Z: {em_target.position[2]:.2f}m<br>" +
                                f"频率: {em_target.frequency/1e9:.2f} GHz<br>" +
                                f"功率: {em_target.tx_power:.1f} dBm"
                            )

                fig.add_trace(go.Scatter3d(
                    x=true_positions[:, 0],
                    y=true_positions[:, 1],
                    z=true_positions[:, 2],
                    mode='markers+text',
                    marker=dict(
                        size=10,
                        color='green',
                        symbol='circle',
                        line=dict(width=2, color='darkgreen')
                    ),
                    text=true_labels,
                    textposition='top center',
                    name='真实位置',
                    showlegend=True,
                    hovertext=true_hover_text,
                    hoverinfo='text'
                ))

            # 添加预测位置（按信号类型着色，使用叉号标记）
            if len(predicted_positions) > 0:
                # 准备自定义悬停数据
                predicted_hover_text = []
                for device in devices:
                    if device.position is not None:
                        error_str = ""
                        if isinstance(self.signal_collector, UniversalEMSignalCollector):
                            if device.mac in self.signal_collector.em_targets:
                                em_target = self.signal_collector.em_targets[device.mac]
                                error = np.linalg.norm(device.position - em_target.position)
                                error_str = f"<br>定位误差: {error:.2f}m"

                        predicted_hover_text.append(
                            f"<b>{device.signal_type} - {device.mac[-8:]}</b><br>" +
                            f"<b>预测位置:</b><br>" +
                            f"X: {device.position[0]:.2f}m<br>" +
                            f"Y: {device.position[1]:.2f}m<br>" +
                            f"Z: {device.position[2]:.2f}m<br>" +
                            f"置信度: {device.confidence:.3f}" +
                            error_str
                        )

                fig.add_trace(go.Scatter3d(
                    x=predicted_positions[:, 0],
                    y=predicted_positions[:, 1],
                    z=predicted_positions[:, 2],
                    mode='markers+text',
                    marker=dict(
                        size=8,
                        color=predicted_colors,
                        symbol='x',  # 使用叉号标记区分
                        line=dict(width=2, color='white')
                    ),
                    text=predicted_labels,
                    textposition='bottom center',
                    name='预测位置',
                    showlegend=True,
                    hovertext=predicted_hover_text,
                    hoverinfo='text'
                ))

            # 添加误差连线（红色虚线）
            for error_line in error_lines:
                pred = error_line['predicted']
                true = error_line['true']
                fig.add_trace(go.Scatter3d(
                    x=[true[0], pred[0]],
                    y=[true[1], pred[1]],
                    z=[true[2], pred[2]],
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name='定位误差',
                    showlegend=False,
                    hoverinfo='skip'
                ))

            fig.update_layout(
                title="实时电磁信号源位置地图",
                scene=dict(
                    xaxis_title="X (米)",
                    yaxis_title="Y (米)",
                    zaxis_title="Z (米)",
                    aspectmode='data',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                ),
                showlegend=True,
                hovermode='closest',
                width=1200,
                height=900
            )

            save_path = os.path.join(PATHS['results'], 'realtime_map.html')
            fig.write_html(save_path)

            self.log(f"实时地图已保存到: {save_path}")
            self.log(f"  显示了 {len(devices)} 个信号源")
            if len(true_positions) > 0:
                self.log(f"  绿色圆圈: {len(true_positions)} 个真实位置")
                self.log(f"  彩色叉号: {len(predicted_positions)} 个预测位置")
                self.log(f"  红色虚线: 定位误差连线")
            if self.model is not None:
                self.log("  包含3D室内模型")

            # 在浏览器中打开
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(save_path))

        except Exception as e:
            self.log(f"错误: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"生成地图失败:\n{str(e)}")

    def _on_closing(self):
        """窗口关闭时的处理"""
        # 停止跟踪
        if self.device_tracker is not None and self.tracking_active:
            self.device_tracker.stop_tracking()

        # 保存设置
        self._save_settings()
        # 关闭窗口
        self.root.destroy()


def main():
    """主函数"""
    print("=" * 60)
    print("启动 GUI...")
    print("=" * 60)

    root = tk.Tk()
    app = IndoorLocalizationGUI(root)

    # 设置窗口关闭处理
    root.protocol("WM_DELETE_WINDOW", app._on_closing)

    print("GUI 初始化完成，窗口已打开")
    print("提示：日志会显示在 GUI 窗口底部的'系统日志'区域")
    print("=" * 60)

    # 不重定向标准输出，这样控制台和GUI都能看到
    # 注释掉原来的重定向代码
    # class StdoutRedirector:
    #     def __init__(self, text_widget):
    #         self.text_widget = text_widget
    #
    #     def write(self, message):
    #         if message.strip():
    #             self.text_widget.log(message.strip())
    #
    #     def flush(self):
    #         pass
    #
    # sys.stdout = StdoutRedirector(app)

    root.mainloop()


if __name__ == "__main__":
    main()
