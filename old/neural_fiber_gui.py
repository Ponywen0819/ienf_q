#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
神經纖維連接實驗GUI工具 - 入口文件

本文件現在只是一個簡單的入口，實際的GUI實現已重構到 gui 資料夾中
"""

import tkinter as tk
from gui import NeuralFiberGUI


def main():
    """主程式"""
    root = tk.Tk()
    app = NeuralFiberGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()