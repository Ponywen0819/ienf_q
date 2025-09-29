#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
神經纖維連接實驗GUI工具 - 主要界面
使用性能優化的視窗區域處理，大幅提升大圖和高縮放比例下的性能
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import sys

from .viewport_processor import ViewportProcessor

# Add the parent directory to the path to import node extraction utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from edge_linking.node_extraction_utils import (
        extract_nodes_from_cropped_region,
        filter_nodes_by_distance,
        get_node_positions,
        print_node_summary
    )
    from edge_linking.edge_connection_utils import (
        connect_neural_fiber_nodes
    )
except ImportError as e:
    print(f"Warning: Could not import edge linking utils: {e}")
    # Define dummy functions to prevent errors
    def extract_nodes_from_cropped_region(*args, **kwargs):
        return []
    def filter_nodes_by_distance(*args, **kwargs):
        return []
    def get_node_positions(*args, **kwargs):
        return []
    def print_node_summary(*args, **kwargs):
        pass
    def connect_neural_fiber_nodes(*args, **kwargs):
        return {'optimal_connections': {}, 'visualization_data': {'edges': [], 'nodes': []}, 'statistics': {}}


class NeuralFiberGUI:
    """神經纖維分析GUI工具 - 性能優化版"""

    def __init__(self, root):
        self.root = root
        self.root.title("神經纖維連接實驗分析工具")
        self.root.geometry("1600x900")

        # 核心處理器
        self.processor = ViewportProcessor()

        # UI狀態
        self._init_ui_state()

        # 建立界面
        self._setup_styles()
        self._setup_ui()
        self._setup_initial_state()

    def _init_ui_state(self):
        """初始化UI狀態"""
        self.current_tool = "drag"
        self.tool_buttons = {}
        self.opacity_buttons = {}
        self.center_opacity_buttons = {}

        # 交互狀態
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.cursor_x = 0
        self.cursor_y = 0

        # 顯示相關
        self.photo = None
        self.canvas_width = 800
        self.canvas_height = 600

        # 節點提取相關
        self.extracted_nodes = []
        self.filtered_nodes = []
        self.node_extraction_thread = None
        self.extraction_cancelled = False
        self.progress_window = None

        # 連線相關
        self.connection_result = None
        self.connection_thread = None
        self.connection_cancelled = False
        self.show_connections = True

    def _setup_styles(self):
        """設定樣式"""
        style = ttk.Style()
        style.configure("Selected.TButton", background="lightblue", relief="sunken")

    def _setup_ui(self):
        """建立界面"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._create_control_panel(main_frame)
        self._create_display_area(main_frame)
        self._create_status_bar()

    def _create_control_panel(self, parent):
        """創建控制面板"""
        # 外層框架
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="5", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # 創建滾動區域
        self._create_scrollable_control_area(control_frame)

    def _create_scrollable_control_area(self, parent):
        """創建可滾動的控制區域"""
        # 創建容器框架
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)

        # 創建畫布和滾動條
        self.control_canvas = tk.Canvas(container, highlightthickness=0, width=270)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.control_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.control_canvas)

        # 配置畫布寬度以適應內容
        def configure_scroll_region(event):
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
            # 確保畫布寬度適應內容
            canvas_width = self.control_canvas.winfo_width()
            self.control_canvas.itemconfig(self.canvas_window, width=canvas_width-20)

        def configure_canvas(event):
            # 當畫布大小改變時，調整內部框架寬度
            canvas_width = event.width
            self.control_canvas.itemconfig(self.canvas_window, width=canvas_width-20)

        self.scrollable_frame.bind("<Configure>", configure_scroll_region)
        self.control_canvas.bind("<Configure>", configure_canvas)

        # 創建畫布中的視窗
        self.canvas_window = self.control_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.control_canvas.configure(yscrollcommand=scrollbar.set)

        # 佈局
        self.control_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 綁定滑鼠滾輪事件
        self._bind_mousewheel_to_canvas()

        # 在可滾動框架中創建各個區塊
        self._create_file_section(self.scrollable_frame)
        self._create_display_section(self.scrollable_frame)
        self._create_tool_section(self.scrollable_frame)
        self._create_info_section(self.scrollable_frame)
        self._create_action_section(self.scrollable_frame)

    def _bind_mousewheel_to_canvas(self):
        """綁定滑鼠滾輪事件到控制面板畫布"""
        def _on_mousewheel(event):
            try:
                if hasattr(event, 'delta') and event.delta:
                    # Windows 和 macOS
                    delta = event.delta
                    self.control_canvas.yview_scroll(int(-1*(delta/120)), "units")
                elif hasattr(event, 'num'):
                    # Linux
                    if event.num == 4:
                        self.control_canvas.yview_scroll(-1, "units")
                    elif event.num == 5:
                        self.control_canvas.yview_scroll(1, "units")
            except Exception as e:
                print(f"Scroll error: {e}")

        def _bind_mouse_events(widget):
            """遞歸綁定滑鼠事件到所有子組件"""
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", _on_mousewheel)
            widget.bind("<Button-5>", _on_mousewheel)

            # 遞歸綁定到所有子組件
            for child in widget.winfo_children():
                _bind_mouse_events(child)

        # 綁定到畫布和所有子組件
        _bind_mouse_events(self.control_canvas)
        _bind_mouse_events(self.scrollable_frame)

        # 延遲綁定，確保所有組件都已創建
        def delayed_bind():
            try:
                _bind_mouse_events(self.scrollable_frame)
            except:
                pass

        self.root.after(100, delayed_bind)

    def _create_file_section(self, parent):
        """文件載入區域"""
        file_frame = ttk.LabelFrame(parent, text="檔案載入")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="載入原始圖片",
                  command=lambda: self._load_image("original"), width=18).pack(pady=2)
        ttk.Button(file_frame, text="載入神經亮點圖片",
                  command=lambda: self._load_image("neural"), width=18).pack(pady=2)
        ttk.Button(file_frame, text="載入Center Line",
                  command=lambda: self._load_image("center"), width=18).pack(pady=2)

    def _create_display_section(self, parent):
        """顯示控制區域"""
        display_frame = ttk.LabelFrame(parent, text="顯示控制")
        display_frame.pack(fill=tk.X, pady=(0, 10))

        # 顯示開關
        self.original_var = tk.BooleanVar(value=True)
        self.neural_var = tk.BooleanVar(value=True)
        self.center_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(display_frame, text="顯示原始圖片",
                       variable=self.original_var, command=self._update_display).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="顯示神經亮點",
                       variable=self.neural_var, command=self._update_display).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="顯示Center Line",
                       variable=self.center_var, command=self._update_display).pack(anchor=tk.W)

        # 通道選擇
        ttk.Label(display_frame, text="顯示通道:").pack(anchor=tk.W, pady=(10, 0))
        self.channel_var = tk.StringVar(value="all")
        channel_combo = ttk.Combobox(display_frame, textvariable=self.channel_var,
                                   values=["all", "red", "green", "blue"],
                                   state="readonly", width=18)
        channel_combo.pack(fill=tk.X, pady=2)
        channel_combo.bind('<<ComboboxSelected>>', lambda e: self._update_display())

        # 透明度控制
        self._create_opacity_controls(display_frame)

    def _create_opacity_controls(self, parent):
        """透明度控制"""
        # 神經亮點透明度
        ttk.Label(parent, text="神經亮點透明度:").pack(anchor=tk.W, pady=(10, 0))
        opacity_frame = ttk.Frame(parent)
        opacity_frame.pack(fill=tk.X, pady=2)

        opacity_values = [0.1, 0.25, 0.5, 0.75]
        for value in opacity_values:
            btn = ttk.Button(opacity_frame, text=f"{int(value*100)}%", width=18,
                           command=lambda v=value: self._set_neural_opacity(v))
            btn.pack(fill=tk.X, pady=1)
            self.opacity_buttons[value] = btn

        # Center Line透明度
        ttk.Label(parent, text="Center Line透明度:").pack(anchor=tk.W, pady=(10, 0))
        center_opacity_frame = ttk.Frame(parent)
        center_opacity_frame.pack(fill=tk.X, pady=2)

        for value in opacity_values:
            btn = ttk.Button(center_opacity_frame, text=f"{int(value*100)}%", width=18,
                           command=lambda v=value: self._set_center_opacity(v))
            btn.pack(fill=tk.X, pady=1)
            self.center_opacity_buttons[value] = btn

    def _create_tool_section(self, parent):
        """工具選擇區域"""
        tool_frame = ttk.LabelFrame(parent, text="工具選擇")
        tool_frame.pack(fill=tk.X, pady=(0, 10))

        self.tool_buttons["drag"] = ttk.Button(tool_frame, text="拖動",
                                              command=lambda: self._set_tool("drag"), width=18)
        self.tool_buttons["drag"].pack(pady=1)

        self.tool_buttons["zoom_in"] = ttk.Button(tool_frame, text="放大",
                                                 command=lambda: self._set_tool("zoom_in"), width=18)
        self.tool_buttons["zoom_in"].pack(pady=1)

        self.tool_buttons["zoom_out"] = ttk.Button(tool_frame, text="縮小",
                                                  command=lambda: self._set_tool("zoom_out"), width=18)
        self.tool_buttons["zoom_out"].pack(pady=1)

        ttk.Button(tool_frame, text="重設縮放",
                  command=self._reset_zoom, width=18).pack(pady=(5, 1))

        # 自定義游標
        self.custom_cursor_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tool_frame, text="顯示自定義游標",
                       variable=self.custom_cursor_var,
                       command=self._toggle_custom_cursor).pack(pady=2)

        # 縮放比例和視窗資訊
        self.zoom_label = ttk.Label(tool_frame, text="縮放: 100%")
        self.zoom_label.pack(pady=(5, 0))

        self.viewport_label = ttk.Label(tool_frame, text="視窗: (0, 0)")
        self.viewport_label.pack(pady=(2, 0))

    def _create_info_section(self, parent):
        """圖片資訊區域"""
        info_frame = ttk.LabelFrame(parent, text="圖片資訊")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_text = tk.Text(info_frame, height=6, width=22, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)

    def _create_action_section(self, parent):
        """操作按鈕區域"""
        action_frame = ttk.LabelFrame(parent, text="工具")
        action_frame.pack(fill=tk.X)

        ttk.Button(action_frame, text="儲存目前視圖",
                  command=self._save_view, width=18).pack(pady=2)
        ttk.Button(action_frame, text="清除所有圖片",
                  command=self._clear_all, width=18).pack(pady=2)

        # 節點提取功能
        ttk.Separator(action_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        ttk.Button(action_frame, text="提取節點",
                  command=self._extract_nodes, width=18).pack(pady=2)

        # 節點提取參數
        params_frame = ttk.Frame(action_frame)
        params_frame.pack(fill=tk.X, pady=2)

        ttk.Label(params_frame, text="最小距離:").pack(anchor=tk.W)
        self.min_distance_var = tk.StringVar(value="5.0")
        ttk.Entry(params_frame, textvariable=self.min_distance_var, width=18).pack(fill=tk.X, pady=1)

        ttk.Label(params_frame, text="最小組件大小:").pack(anchor=tk.W, pady=(5, 0))
        self.min_component_size_var = tk.StringVar(value="10")
        ttk.Entry(params_frame, textvariable=self.min_component_size_var, width=18).pack(fill=tk.X, pady=1)

        # 連線功能
        ttk.Separator(action_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        ttk.Button(action_frame, text="計算連線",
                  command=self._calculate_connections, width=18).pack(pady=2)

        # 連線參數
        conn_params_frame = ttk.Frame(action_frame)
        conn_params_frame.pack(fill=tk.X, pady=2)

        ttk.Label(conn_params_frame, text="最大距離:").pack(anchor=tk.W)
        self.max_distance_var = tk.StringVar(value="20.0")
        ttk.Entry(conn_params_frame, textvariable=self.max_distance_var, width=18).pack(fill=tk.X, pady=1)

        ttk.Label(conn_params_frame, text="算法:").pack(anchor=tk.W, pady=(5, 0))
        self.algorithm_var = tk.StringVar(value="astar")
        algorithm_combo = ttk.Combobox(conn_params_frame, textvariable=self.algorithm_var,
                                     values=["astar", "dijkstra"], state="readonly", width=18)
        algorithm_combo.pack(fill=tk.X, pady=1)

        ttk.Label(conn_params_frame, text="策略:").pack(anchor=tk.W, pady=(5, 0))
        self.strategy_var = tk.StringVar(value="nearest_neighbor")
        strategy_combo = ttk.Combobox(conn_params_frame, textvariable=self.strategy_var,
                                    values=["nearest_neighbor", "minimum_spanning_tree", "all_pairs"],
                                    state="readonly", width=18)
        strategy_combo.pack(fill=tk.X, pady=1)

        # 連線顯示控制
        self.show_connections_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(conn_params_frame, text="顯示連線",
                       variable=self.show_connections_var,
                       command=self._toggle_connections_display).pack(anchor=tk.W, pady=(5, 0))

    def _create_display_area(self, parent):
        """創建顯示區域"""
        display_frame = ttk.LabelFrame(parent, text="圖片顯示")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(display_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 畫布（不需要滾動條，因為我們直接控制視窗）
        self.canvas = tk.Canvas(canvas_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 綁定事件
        self._bind_canvas_events()

        # 綁定畫布尺寸變更事件
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _create_status_bar(self):
        """創建狀態列"""
        self.status_var = tk.StringVar(value="準備就緒")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_canvas_events(self):
        """綁定畫布事件"""
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)  # Linux
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)  # Linux
        self.canvas.bind("<Button-1>", self._on_mouse_click)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<Motion>", self._on_mouse_motion)
        self.canvas.bind("<Enter>", self._on_canvas_enter)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

    def _setup_initial_state(self):
        """設定初始狀態"""
        self._set_tool("drag")
        self._update_opacity_buttons()

    def _on_canvas_resize(self, event):
        """畫布尺寸改變"""
        self.canvas_width = event.width
        self.canvas_height = event.height

        # 更新處理器的視窗尺寸
        self.processor.set_viewport(
            self.processor.viewport_x,
            self.processor.viewport_y,
            self.canvas_width,
            self.canvas_height,
            self.processor.zoom_factor
        )
        self._update_display()

    # === 事件處理方法 ===
    def _on_mouse_wheel(self, event):
        """滑鼠滾輪縮放"""
        if not self.processor._has_any_image():
            return

        self._update_cursor_position(event.x, event.y)
        zoom_step = 1.1 if abs(event.delta) < 100 else 1.2

        zoom_ratio = zoom_step if (event.delta > 0 or event.num == 4) else (1/zoom_step)

        if self.processor.zoom_at_point(zoom_ratio, self.cursor_x, self.cursor_y):
            self._update_display()
            self._update_viewport_info()

    def _on_mouse_click(self, event):
        """滑鼠點擊"""
        self._update_cursor_position(event.x, event.y)

        if self.current_tool == "drag":
            self._start_drag(event)
        elif self.current_tool == "zoom_in":
            if self.processor.zoom_at_point(1.2, self.cursor_x, self.cursor_y):
                self._update_display()
                self._update_viewport_info()
        elif self.current_tool == "zoom_out":
            if self.processor.zoom_at_point(1/1.2, self.cursor_x, self.cursor_y):
                self._update_display()
                self._update_viewport_info()

    def _on_mouse_drag(self, event):
        """滑鼠拖動"""
        if self.current_tool == "drag" and self.is_dragging:
            dx = self.drag_start_x - event.x
            dy = self.drag_start_y - event.y

            self.processor.pan_viewport(dx, dy)
            self._update_display()
            self._update_viewport_info()

            self.drag_start_x = event.x
            self.drag_start_y = event.y

    def _on_mouse_release(self, event):
        """滑鼠釋放"""
        if self.current_tool == "drag":
            self._end_drag()

    def _on_mouse_motion(self, event):
        """滑鼠移動"""
        self._update_cursor_position(event.x, event.y)
        if self.processor._has_any_image() and self.custom_cursor_var.get():
            self._update_custom_cursor()

    def _on_canvas_enter(self, event):
        """進入畫布"""
        if self.custom_cursor_var.get():
            self.canvas.configure(cursor="none")

    def _on_canvas_leave(self, event):
        """離開畫布"""
        self._remove_custom_cursor()

    # === 圖片載入方法 ===
    def _load_image(self, image_type):
        """載入圖片"""
        type_names = {
            "original": "原始圖片",
            "neural": "神經亮點圖片",
            "center": "Center Line圖片"
        }

        file_path = filedialog.askopenfilename(
            title=f"選擇{type_names[image_type]}",
            filetypes=[("圖片檔案", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
                      ("所有檔案", "*.*")]
        )

        if not file_path:
            return

        try:
            img = Image.open(file_path)
            img_array = np.array(img)

            # 確保RGB格式
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif len(img_array.shape) == 3 and img_array.shape[2] > 3:
                img_array = img_array[:, :, :3]

            # 設定到處理器
            kwargs = {f"{image_type}": img_array}
            self.processor.set_images(**kwargs)

            # 重設視窗到圖片開始位置
            self._reset_viewport()

            self._update_display()
            self._update_info()
            self._update_viewport_info()
            self.status_var.set(f"已載入{type_names[image_type]}: {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("錯誤", f"載入圖片失敗: {str(e)}")

    def _reset_viewport(self):
        """重設視窗到圖片開始位置"""
        self.processor.set_viewport(0, 0, self.canvas_width, self.canvas_height, 1.0)

    # === 顯示更新方法 ===
    def _update_display(self):
        """更新顯示"""
        # 更新處理器的顯示設定
        self.processor.set_display_settings(
            show_original=self.original_var.get(),
            show_neural=self.neural_var.get(),
            show_center=self.center_var.get(),
            channel=self.channel_var.get(),
            neural_opacity=self.processor.neural_opacity,
            center_opacity=self.processor.center_opacity
        )

        # 更新視窗尺寸
        self.processor.set_viewport(
            self.processor.viewport_x,
            self.processor.viewport_y,
            self.canvas_width,
            self.canvas_height,
            self.processor.zoom_factor
        )

        # 獲取處理後的視窗圖片
        viewport_img = self.processor.get_viewport_image()

        if viewport_img is None:
            self.canvas.delete("all")
            return

        # 轉換為Tkinter可顯示的格式
        pil_image = Image.fromarray(viewport_img)
        self.photo = ImageTk.PhotoImage(pil_image)

        # 顯示圖片（填滿畫布）
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # 如果有節點，重新繪製它們
        if hasattr(self, 'filtered_nodes') and self.filtered_nodes:
            self._draw_nodes_on_canvas()

        # 如果有連線結果且設定為顯示，重新繪製連線
        if (hasattr(self, 'connection_result') and self.connection_result and
            hasattr(self, 'show_connections_var') and self.show_connections_var.get()):
            self._draw_connections_on_canvas()

        self._update_zoom_label()

    def _update_zoom_label(self):
        """更新縮放標籤"""
        percentage = int(self.processor.zoom_factor * 100)
        self.zoom_label.config(text=f"縮放: {percentage}%")

    def _update_viewport_info(self):
        """更新視窗資訊"""
        x, y = self.processor.viewport_x, self.processor.viewport_y
        self.viewport_label.config(text=f"視窗: ({x}, {y})")

    def _update_info(self):
        """更新圖片資訊"""
        info_text = ""
        img_width, img_height = self.processor.get_full_image_size()

        if img_width > 0 and img_height > 0:
            info_text += f"圖片尺寸: {img_width} x {img_height}\n"
            info_text += f"視窗尺寸: {self.canvas_width} x {self.canvas_height}\n"
            info_text += f"縮放係數: {self.processor.zoom_factor:.2f}\n"

            # 計算當前處理的區域大小
            crop_w = int(self.canvas_width / self.processor.zoom_factor)
            crop_h = int(self.canvas_height / self.processor.zoom_factor)
            total_pixels = img_width * img_height
            visible_pixels = crop_w * crop_h
            efficiency = (visible_pixels / total_pixels) * 100 if total_pixels > 0 else 0

            info_text += f"處理區域: {crop_w} x {crop_h}\n"
            info_text += f"處理效率: {efficiency:.1f}%\n"

        for img, name in [(self.processor.original_image, "原始圖片"),
                         (self.processor.neural_spots_image, "神經亮點"),
                         (self.processor.center_line_image, "Center Line")]:
            if img is not None:
                info_text += f"\n{name}: 已載入\n"

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)

    # === 工具控制方法 ===
    def _set_tool(self, tool_name):
        """設定工具"""
        self.current_tool = tool_name

        for name, button in self.tool_buttons.items():
            style = "Selected.TButton" if name == tool_name else "TButton"
            button.configure(style=style)

        if self.custom_cursor_var.get():
            self.canvas.configure(cursor="none")
        else:
            cursor_map = {"drag": "fleur", "zoom_in": "crosshair", "zoom_out": "dotbox"}
            self.canvas.configure(cursor=cursor_map.get(tool_name, "arrow"))

    def _set_neural_opacity(self, value):
        """設定神經亮點透明度"""
        self.processor.set_display_settings(neural_opacity=value)
        self._update_opacity_buttons()
        self._update_display()

    def _set_center_opacity(self, value):
        """設定Center Line透明度"""
        self.processor.set_display_settings(center_opacity=value)
        self._update_center_opacity_buttons()
        self._update_display()

    def _update_opacity_buttons(self):
        """更新透明度按鈕狀態"""
        for value, button in self.opacity_buttons.items():
            style = "Selected.TButton" if abs(value - self.processor.neural_opacity) < 0.01 else "TButton"
            button.configure(style=style)

    def _update_center_opacity_buttons(self):
        """更新Center Line透明度按鈕狀態"""
        for value, button in self.center_opacity_buttons.items():
            style = "Selected.TButton" if abs(value - self.processor.center_opacity) < 0.01 else "TButton"
            button.configure(style=style)

    def _reset_zoom(self):
        """重設縮放"""
        self.processor.set_viewport(0, 0, self.canvas_width, self.canvas_height, 1.0)
        self._update_display()
        self._update_viewport_info()

    # === 游標相關方法 ===
    def _update_cursor_position(self, x, y):
        """更新游標位置"""
        self.cursor_x, self.cursor_y = x, y

    def _update_custom_cursor(self):
        """更新自定義游標"""
        if not self.custom_cursor_var.get():
            return

        self.canvas.delete("custom_cursor")

        size, width = 20, 2
        x, y = self.cursor_x, self.cursor_y

        # 黑色邊框
        self.canvas.create_line(x, y-size-1, x, y+size+1,
                               fill="black", width=width+2, tags="custom_cursor")
        self.canvas.create_line(x-size-1, y, x+size+1, y,
                               fill="black", width=width+2, tags="custom_cursor")

        # 白色前景
        self.canvas.create_line(x, y-size, x, y+size,
                               fill="white", width=width, tags="custom_cursor")
        self.canvas.create_line(x-size, y, x+size, y,
                               fill="white", width=width, tags="custom_cursor")

        # 紅色中心點
        self.canvas.create_oval(x-3, y-3, x+3, y+3,
                               fill="red", outline="white", width=1, tags="custom_cursor")

    def _remove_custom_cursor(self):
        """移除自定義游標"""
        self.canvas.delete("custom_cursor")

    def _toggle_custom_cursor(self):
        """切換自定義游標"""
        if not self.custom_cursor_var.get():
            self._remove_custom_cursor()
        self._set_tool(self.current_tool)

    # === 拖動相關方法 ===
    def _start_drag(self, event):
        """開始拖動"""
        self.drag_start_x, self.drag_start_y = event.x, event.y
        self.is_dragging = True
        self.canvas.configure(cursor="fleur")

    def _end_drag(self):
        """結束拖動"""
        self.is_dragging = False
        self._set_tool(self.current_tool)

    # === 其他功能方法 ===
    def _save_view(self):
        """儲存當前視圖"""
        viewport_img = self.processor.get_viewport_image()
        if viewport_img is None:
            messagebox.showwarning("警告", "沒有圖片可以儲存")
            return

        file_path = filedialog.asksaveasfilename(
            title="儲存圖片",
            defaultextension=".png",
            filetypes=[("PNG檔案", "*.png"), ("JPEG檔案", "*.jpg"), ("所有檔案", "*.*")]
        )

        if file_path:
            try:
                pil_image = Image.fromarray(viewport_img)
                pil_image.save(file_path)
                self.status_var.set(f"已儲存: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("錯誤", f"儲存失敗: {str(e)}")

    def _clear_all(self):
        """清除所有圖片"""
        self.processor.set_images(original=None, neural=None, center=None)
        self.canvas.delete("all")
        self.info_text.delete(1.0, tk.END)
        self.status_var.set("已清除所有圖片")
        # 清除節點資料
        self.extracted_nodes = []
        self.filtered_nodes = []
        # 清除連線資料
        self.connection_result = None

    # === 節點提取相關方法 ===
    def _check_data_for_node_extraction(self):
        """檢查節點提取所需的資料是否已載入"""
        required_data = []
        missing_data = []

        # 檢查必要的圖片
        if self.processor.original_image is not None:
            required_data.append("原始圖片")
        else:
            missing_data.append("原始圖片")

        if self.processor.neural_spots_image is not None:
            required_data.append("神經亮點圖片")
        else:
            missing_data.append("神經亮點圖片")

        # Center Line 是可選的，但建議有
        center_optional = self.processor.center_line_image is not None

        return {
            'has_required': len(missing_data) == 0,
            'required_data': required_data,
            'missing_data': missing_data,
            'has_center': center_optional
        }

    def _extract_nodes(self):
        """開始節點提取過程"""
        # 檢查資料
        data_check = self._check_data_for_node_extraction()

        if not data_check['has_required']:
            missing_str = "、".join(data_check['missing_data'])
            messagebox.showerror(
                "資料不足",
                f"節點提取需要以下資料：\n\n缺少: {missing_str}\n\n請先載入所需的圖片。"
            )
            return

        # 顯示確認對話框
        confirm_msg = f"已載入: {', '.join(data_check['required_data'])}"
        if data_check['has_center']:
            confirm_msg += "\n已載入 Center Line (可選)"
        else:
            confirm_msg += "\n未載入 Center Line (可選，建議載入)"

        confirm_msg += "\n\n是否要開始節點提取？\n注意：\n- 將對整張圖片進行節點提取\n- 提取過程中將無法操作界面\n- 大圖片可能需要較長時間"

        if not messagebox.askyesno("確認節點提取", confirm_msg):
            return

        # 開始節點提取
        self._start_node_extraction()

    def _start_node_extraction(self):
        """開始節點提取（在背景執行緒中）"""
        # 取得參數
        try:
            min_distance = float(self.min_distance_var.get())
            min_component_size = int(self.min_component_size_var.get())
        except ValueError:
            messagebox.showerror("參數錯誤", "請輸入有效的數值參數")
            return

        # 重置取消狀態
        self.extraction_cancelled = False

        # 顯示進度視窗
        self._show_progress_window()

        # 在背景執行緒中執行節點提取
        self.node_extraction_thread = threading.Thread(
            target=self._perform_node_extraction,
            args=(min_distance, min_component_size),
            daemon=True
        )
        self.node_extraction_thread.start()

    def _show_progress_window(self):
        """顯示進度視窗"""
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("節點提取進行中")
        self.progress_window.geometry("400x200")
        self.progress_window.resizable(False, False)

        # 設為模態視窗
        self.progress_window.transient(self.root)
        self.progress_window.grab_set()

        # 置中顯示
        self.progress_window.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 200,
            self.root.winfo_rooty() + 150
        ))

        # 內容
        main_frame = ttk.Frame(self.progress_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="正在提取節點...",
                 font=("Arial", 12)).pack(pady=(0, 10))

        # 進度條
        self.progress_bar = ttk.Progressbar(
            main_frame, mode='indeterminate', length=300
        )
        self.progress_bar.pack(pady=(0, 15))
        self.progress_bar.start(10)

        # 狀態標籤
        self.progress_status = ttk.Label(main_frame, text="正在分析圖片...")
        self.progress_status.pack(pady=(0, 15))

        # 取消按鈕
        ttk.Button(main_frame, text="取消",
                  command=self._cancel_extraction).pack()

        # 防止關閉視窗
        self.progress_window.protocol("WM_DELETE_WINDOW", self._cancel_extraction)

    def _cancel_extraction(self):
        """取消節點提取"""
        self.extraction_cancelled = True
        self._hide_progress_window()
        self.status_var.set("節點提取已取消")

    def _hide_progress_window(self):
        """隱藏進度視窗"""
        if self.progress_window:
            self.progress_bar.stop()
            self.progress_window.grab_release()
            self.progress_window.destroy()
            self.progress_window = None

    def _update_progress_status(self, message):
        """更新進度狀態（在主執行緒中調用）"""
        if self.progress_status and not self.extraction_cancelled:
            self.root.after(0, lambda: self.progress_status.config(text=message))

    def _perform_node_extraction(self, min_distance, min_component_size):
        """執行節點提取（在背景執行緒中）"""
        try:
            if self.extraction_cancelled:
                return

            # 更新狀態
            self._update_progress_status("正在準備資料...")

            if self.extraction_cancelled:
                return

            # 更新狀態
            self._update_progress_status("正在分析連通組件...")

            # 使用神經亮點圖片作為標籤圖片進行節點提取
            input_image = self.processor.original_image
            raw_label_image = self.processor.neural_spots_image
            raw_mask_image = self.processor.center_line_image  # 可選的遮罩

            # 確保標籤圖片是單通道
            if len(raw_label_image.shape) == 3:
                # 轉換為灰階
                if raw_label_image.shape[2] == 3:
                    label_image = np.dot(raw_label_image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                else:
                    label_image = raw_label_image[:, :, 0].astype(np.uint8)
            else:
                label_image = raw_label_image.astype(np.uint8)

            # 確保遮罩圖片也是單通道（如果存在）
            mask_image = None
            if raw_mask_image is not None:
                if len(raw_mask_image.shape) == 3:
                    if raw_mask_image.shape[2] == 3:
                        mask_image = np.dot(raw_mask_image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                    else:
                        mask_image = raw_mask_image[:, :, 0].astype(np.uint8)
                else:
                    mask_image = raw_mask_image.astype(np.uint8)

            # 由於要處理整張圖片，直接使用基本的節點提取函數
            from edge_linking.node_extraction_utils import extract_nodes_from_label_image

            nodes = extract_nodes_from_label_image(
                input_image=input_image,
                label_image=label_image,
                mask_image=mask_image,
                min_component_size=min_component_size
            )

            print(f"提取到 {len(nodes)} 個節點")

            if self.extraction_cancelled:
                return

            # 更新狀態
            self._update_progress_status("正在過濾節點...")

            # 過濾節點
            filtered_nodes = filter_nodes_by_distance(nodes, min_distance=min_distance)

            if self.extraction_cancelled:
                return

            # 在主執行緒中更新結果
            self.root.after(0, lambda: self._extraction_completed(nodes, filtered_nodes))

        except Exception as e:
            # 在主執行緒中顯示錯誤
            error_msg = f"節點提取過程中發生錯誤：\n{str(e)}"
            self.root.after(0, lambda: self._extraction_error(error_msg))

    def _extraction_completed(self, nodes, filtered_nodes):
        """節點提取完成（在主執行緒中調用）"""
        self._hide_progress_window()

        # 儲存結果
        self.extracted_nodes = nodes
        self.filtered_nodes = filtered_nodes

        # 顯示結果
        result_msg = f"節點提取完成！\n\n"
        result_msg += f"原始節點數量: {len(nodes)}\n"
        result_msg += f"過濾後節點數量: {len(filtered_nodes)}\n\n"

        if filtered_nodes:
            result_msg += "節點資訊:\n"
            for i, node in enumerate(filtered_nodes[:5]):  # 只顯示前5個
                result_msg += f"節點 {i+1}: 位置({node['position'][0]}, {node['position'][1]}), "
                result_msg += f"像素值={node['pixel_value']}\n"
            if len(filtered_nodes) > 5:
                result_msg += f"... 還有 {len(filtered_nodes)-5} 個節點\n"

        messagebox.showinfo("節點提取結果", result_msg)
        self.status_var.set(f"已提取 {len(filtered_nodes)} 個節點")

        # 更新顯示以顯示節點
        self._update_display_with_nodes()

    def _extraction_error(self, error_msg):
        """節點提取錯誤（在主執行緒中調用）"""
        self._hide_progress_window()
        messagebox.showerror("節點提取錯誤", error_msg)
        self.status_var.set("節點提取失敗")

    def _update_display_with_nodes(self):
        """更新顯示以包含節點標記"""
        # 先正常更新顯示
        self._update_display()

        # 如果有過濾後的節點，在畫布上繪製它們
        if self.filtered_nodes:
            self._draw_nodes_on_canvas()

        # 如果有連線結果且設定為顯示，繪製連線
        if (hasattr(self, 'connection_result') and self.connection_result and
            hasattr(self, 'show_connections') and self.show_connections):
            self._draw_connections_on_canvas()

    def _draw_nodes_on_canvas(self):
        """在畫布上繪製節點"""
        if not self.filtered_nodes:
            return

        # 獲取當前視窗的源區域資訊（參考 viewport_processor 的邏輯）
        viewport_x = self.processor.viewport_x
        viewport_y = self.processor.viewport_y
        zoom_factor = self.processor.zoom_factor

        # 計算源區域（原圖中的位置）
        source_x = int(viewport_x / zoom_factor)
        source_y = int(viewport_y / zoom_factor)
        source_width = int(self.canvas_width / zoom_factor)
        source_height = int(self.canvas_height / zoom_factor)

        # 調試輸出
        print(f"繪製節點: viewport=({viewport_x}, {viewport_y}), zoom={zoom_factor:.2f}")
        print(f"源區域: ({source_x}, {source_y}) -> ({source_x + source_width}, {source_y + source_height})")

        visible_nodes = 0
        for i, node in enumerate(self.filtered_nodes):
            node_y, node_x = node['position']

            # 檢查節點是否在當前可見的源區域內
            if (source_x <= node_x <= source_x + source_width and
                source_y <= node_y <= source_y + source_height):

                # 計算節點在裁剪區域中的相對位置
                relative_x = node_x - source_x
                relative_y = node_y - source_y

                # 轉換到畫布座標（按比例縮放到視窗大小）
                canvas_x = (relative_x * self.canvas_width) / source_width
                canvas_y = (relative_y * self.canvas_height) / source_height

                # 調試輸出前幾個節點的座標轉換
                if i < 3:
                    print(f"節點 {i}: 原始位置({node_y}, {node_x}) -> 相對位置({relative_y}, {relative_x}) -> 畫布位置({canvas_x:.1f}, {canvas_y:.1f})")

                visible_nodes += 1
                # 繪製節點標記
                node_size = max(6, min(15, int(12)))  # 固定合適的大小
                self.canvas.create_oval(
                    canvas_x - node_size, canvas_y - node_size,
                    canvas_x + node_size, canvas_y + node_size,
                    fill="yellow", outline="black", width=2,
                    tags="node_marker"
                )

                # 繪製節點編號
                font_size = max(8, min(12, int(10)))  # 固定合適的字體大小
                self.canvas.create_text(
                    canvas_x, canvas_y,
                    text=str(i), fill="black", font=("Arial", font_size, "bold"),
                    tags="node_marker"
                )

        print(f"在當前視窗中顯示了 {visible_nodes} 個節點（總共 {len(self.filtered_nodes)} 個）")

    # === 連線相關方法 ===
    def _check_nodes_for_connection(self):
        """檢查連線所需的節點是否已提取"""
        if not self.filtered_nodes:
            return {
                'has_nodes': False,
                'node_count': 0,
                'message': '尚未提取節點，請先執行「提取節點」功能'
            }

        if len(self.filtered_nodes) < 2:
            return {
                'has_nodes': False,
                'node_count': len(self.filtered_nodes),
                'message': f'至少需要 2 個節點才能進行連線，目前只有 {len(self.filtered_nodes)} 個'
            }

        return {
            'has_nodes': True,
            'node_count': len(self.filtered_nodes),
            'message': f'可以進行連線，共 {len(self.filtered_nodes)} 個節點'
        }

    def _calculate_connections(self):
        """開始連線計算過程"""
        # 檢查節點
        node_check = self._check_nodes_for_connection()

        if not node_check['has_nodes']:
            messagebox.showerror(
                "節點不足",
                node_check['message']
            )
            return

        # 顯示確認對話框
        confirm_msg = f"{node_check['message']}\n\n"
        confirm_msg += f"參數設定：\n"
        confirm_msg += f"- 最大距離: {self.max_distance_var.get()}\n"
        confirm_msg += f"- 算法: {self.algorithm_var.get()}\n"
        confirm_msg += f"- 策略: {self.strategy_var.get()}\n\n"
        confirm_msg += f"是否要開始連線計算？\n注意：\n- 計算過程中將無法操作界面\n- 節點數量較多時可能需要較長時間"

        if not messagebox.askyesno("確認連線計算", confirm_msg):
            return

        # 開始連線計算
        self._start_connection_calculation()

    def _start_connection_calculation(self):
        """開始連線計算（在背景執行緒中）"""
        # 取得參數
        try:
            max_distance = float(self.max_distance_var.get())
            algorithm = self.algorithm_var.get()
            strategy = self.strategy_var.get()
        except ValueError:
            messagebox.showerror("參數錯誤", "請輸入有效的數值參數")
            return

        # 重置取消狀態
        self.connection_cancelled = False

        # 顯示進度視窗
        self._show_connection_progress_window()

        # 在背景執行緒中執行連線計算
        self.connection_thread = threading.Thread(
            target=self._perform_connection_calculation,
            args=(max_distance, algorithm, strategy),
            daemon=True
        )
        self.connection_thread.start()

    def _show_connection_progress_window(self):
        """顯示連線計算進度視窗"""
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("連線計算進行中")
        self.progress_window.geometry("400x200")
        self.progress_window.resizable(False, False)

        # 設為模態視窗
        self.progress_window.transient(self.root)
        self.progress_window.grab_set()

        # 置中顯示
        self.progress_window.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 200,
            self.root.winfo_rooty() + 150
        ))

        # 內容
        main_frame = ttk.Frame(self.progress_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="正在計算節點連線...",
                 font=("Arial", 12)).pack(pady=(0, 10))

        # 進度條
        self.progress_bar = ttk.Progressbar(
            main_frame, mode='indeterminate', length=300
        )
        self.progress_bar.pack(pady=(0, 15))
        self.progress_bar.start(10)

        # 狀態標籤
        self.progress_status = ttk.Label(main_frame, text="正在初始化...")
        self.progress_status.pack(pady=(0, 15))

        # 取消按鈕
        ttk.Button(main_frame, text="取消",
                  command=self._cancel_connection_calculation).pack()

        # 防止關閉視窗
        self.progress_window.protocol("WM_DELETE_WINDOW", self._cancel_connection_calculation)

    def _cancel_connection_calculation(self):
        """取消連線計算"""
        self.connection_cancelled = True
        self._hide_progress_window()
        self.status_var.set("連線計算已取消")

    def _update_connection_progress_status(self, message):
        """更新連線計算進度狀態（在主執行緒中調用）"""
        if self.progress_status and not self.connection_cancelled:
            self.root.after(0, lambda: self.progress_status.config(text=message))

    def _perform_connection_calculation(self, max_distance, algorithm, strategy):
        """執行連線計算（在背景執行緒中）"""
        try:
            if self.connection_cancelled:
                return

            # 更新狀態
            self._update_connection_progress_status("正在準備資料...")

            # 獲取原始圖片用於路徑計算
            input_image = self.processor.original_image
            if len(input_image.shape) == 3:
                # 使用第二個通道（索引 1）
                pathfinding_image = input_image[:, :, 1]
            else:
                pathfinding_image = input_image

            if self.connection_cancelled:
                return

            # 更新狀態
            self._update_connection_progress_status("正在計算節點間連線...")

            # 執行連線計算
            connection_result = connect_neural_fiber_nodes(
                nodes=self.filtered_nodes,
                image=pathfinding_image,
                max_distance=max_distance,
                pathfinding_algorithm=algorithm,
                connection_strategy=strategy
            )

            if self.connection_cancelled:
                return

            # 在主執行緒中更新結果
            self.root.after(0, lambda: self._connection_calculation_completed(connection_result))

        except Exception as e:
            # 在主執行緒中顯示錯誤
            error_msg = f"連線計算過程中發生錯誤：\n{str(e)}"
            self.root.after(0, lambda: self._connection_calculation_error(error_msg))

    def _connection_calculation_completed(self, connection_result):
        """連線計算完成（在主執行緒中調用）"""
        self._hide_progress_window()

        # 儲存結果
        self.connection_result = connection_result

        # 顯示結果
        stats = connection_result['statistics']
        result_msg = f"連線計算完成！\n\n"
        result_msg += f"結果統計：\n"
        result_msg += f"- 節點數量: {stats['total_nodes']}\n"
        result_msg += f"- 潛在連線: {stats['potential_connections']}\n"
        result_msg += f"- 最佳連線: {stats['optimal_connections']}\n"
        result_msg += f"- 使用算法: {stats['algorithm_used']}\n"
        result_msg += f"- 使用策略: {stats['strategy_used']}"

        messagebox.showinfo("連線計算結果", result_msg)
        self.status_var.set(f"已計算 {stats['optimal_connections']} 條連線")

        # 更新顯示以顯示連線
        self._update_display_with_connections()

    def _connection_calculation_error(self, error_msg):
        """連線計算錯誤（在主執行緒中調用）"""
        self._hide_progress_window()
        messagebox.showerror("連線計算錯誤", error_msg)
        self.status_var.set("連線計算失敗")

    def _toggle_connections_display(self):
        """切換連線顯示"""
        self.show_connections = self.show_connections_var.get()
        self._update_display()

    def _update_display_with_connections(self):
        """更新顯示以包含連線標記"""
        # 先正常更新顯示
        self._update_display()

        # 如果有連線結果且設定為顯示，在畫布上繪製它們
        if self.connection_result and self.show_connections:
            self._draw_connections_on_canvas()

    def _draw_connections_on_canvas(self):
        """在畫布上繪製連線"""
        if not self.connection_result or not self.show_connections:
            return

        # 獲取當前視窗的源區域資訊
        viewport_x = self.processor.viewport_x
        viewport_y = self.processor.viewport_y
        zoom_factor = self.processor.zoom_factor

        # 計算源區域（原圖中的位置）
        source_x = int(viewport_x / zoom_factor)
        source_y = int(viewport_y / zoom_factor)
        source_width = int(self.canvas_width / zoom_factor)
        source_height = int(self.canvas_height / zoom_factor)

        viz_data = self.connection_result['visualization_data']

        for edge in viz_data['edges']:
            # 獲取pathfinding路徑
            path = edge.get('path', [])

            if path and len(path) > 1:
                # 繪製pathfinding路徑
                self._draw_pathfinding_path(path, source_x, source_y, source_width, source_height)
            else:
                # 如果沒有路徑資料，回到直線連接
                source_node_id = edge['source']
                target_node_id = edge['target']
                source_pos = self.filtered_nodes[source_node_id]['position']
                target_pos = self.filtered_nodes[target_node_id]['position']

                self._draw_straight_connection(source_pos, target_pos, source_x, source_y, source_width, source_height)

            # 繪製權重標籤
            weight = edge['weight']
            source_node_id = edge['source']
            target_node_id = edge['target']
            source_pos = self.filtered_nodes[source_node_id]['position']
            target_pos = self.filtered_nodes[target_node_id]['position']

            # 計算中點位置用於顯示權重
            mid_pos_y = (source_pos[0] + target_pos[0]) / 2
            mid_pos_x = (source_pos[1] + target_pos[1]) / 2

            # 檢查中點是否在可見範圍內
            if (source_x <= mid_pos_x <= source_x + source_width and
                source_y <= mid_pos_y <= source_y + source_height):

                mid_canvas_x = ((mid_pos_x - source_x) * self.canvas_width) / source_width
                mid_canvas_y = ((mid_pos_y - source_y) * self.canvas_height) / source_height

                self.canvas.create_text(
                    mid_canvas_x, mid_canvas_y,
                    text=f"{weight:.1f}", fill="blue", font=("Arial", 8, "bold"),
                    tags="connection_line"
                )

        print(f"繪製了 {len(viz_data['edges'])} 條連線")

    def _draw_pathfinding_path(self, path, source_x, source_y, source_width, source_height):
        """繪製pathfinding路徑"""
        canvas_points = []

        # 將路徑點轉換為畫布座標
        for i, (y, x) in enumerate(path):
            # 檢查點是否在可見範圍內
            if (source_x <= x <= source_x + source_width and
                source_y <= y <= source_y + source_height):

                canvas_x = ((x - source_x) * self.canvas_width) / source_width
                canvas_y = ((y - source_y) * self.canvas_height) / source_height
                canvas_points.extend([canvas_x, canvas_y])

        # 如果有足夠的點來繪製路徑
        if len(canvas_points) >= 4:  # 至少需要2個點（4個座標值）
            self.canvas.create_line(
                *canvas_points,
                fill="red", width=2, smooth=True,
                tags="connection_line"
            )

    def _draw_straight_connection(self, source_pos, target_pos, source_x, source_y, source_width, source_height):
        """繪製直線連接（備用方案）"""
        # 檢查是否有任一節點在可見範圍內
        source_visible = (source_x <= source_pos[1] <= source_x + source_width and
                        source_y <= source_pos[0] <= source_y + source_height)
        target_visible = (source_x <= target_pos[1] <= source_x + source_width and
                        source_y <= target_pos[0] <= source_y + source_height)

        if source_visible or target_visible:
            # 計算在畫布上的位置
            source_canvas_x = ((source_pos[1] - source_x) * self.canvas_width) / source_width
            source_canvas_y = ((source_pos[0] - source_y) * self.canvas_height) / source_height
            target_canvas_x = ((target_pos[1] - source_x) * self.canvas_width) / source_width
            target_canvas_y = ((target_pos[0] - source_y) * self.canvas_height) / source_height

            # 繪製連線
            self.canvas.create_line(
                source_canvas_x, source_canvas_y,
                target_canvas_x, target_canvas_y,
                fill="orange", width=2, tags="connection_line"
            )