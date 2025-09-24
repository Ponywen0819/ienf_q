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

from .viewport_processor import ViewportProcessor


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
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        self._create_file_section(control_frame)
        self._create_display_section(control_frame)
        self._create_tool_section(control_frame)
        self._create_info_section(control_frame)
        self._create_action_section(control_frame)

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
                                   state="readonly", width=15)
        channel_combo.pack(pady=2)
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
            btn = ttk.Button(opacity_frame, text=f"{int(value*100)}%", width=6,
                           command=lambda v=value: self._set_neural_opacity(v))
            btn.pack(side=tk.LEFT, padx=1)
            self.opacity_buttons[value] = btn

        # Center Line透明度
        ttk.Label(parent, text="Center Line透明度:").pack(anchor=tk.W, pady=(10, 0))
        center_opacity_frame = ttk.Frame(parent)
        center_opacity_frame.pack(fill=tk.X, pady=2)

        for value in opacity_values:
            btn = ttk.Button(center_opacity_frame, text=f"{int(value*100)}%", width=6,
                           command=lambda v=value: self._set_center_opacity(v))
            btn.pack(side=tk.LEFT, padx=1)
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