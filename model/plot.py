"""
plot.py

一个使用 matplotlib 库来绘制简单条形图（bar chart）的脚本。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 


# --- 注释掉的数据加载代码 ---
# (这部分代码被注释掉了，它演示了如何从 Excel 文件加载数据)
# d=pd.read_excel('E:\Python\projects\data\data100.xlsx',header=None)
# d=d[0]
# d=list(d)
# ---

# --- 1. 准备绘图数据 ---
# X 轴数据（0 到 10，代表 11 个类别或区间）
ages = range(11)
# Y 轴数据（每个条形的高度）
count = [0.1, 0.3, 0.2, 0.9, 0.8, 0.9, 0.1, 0.7, 0.8, 0.8, 0.6]

# --- 2. 创建条形图 ---
# 调用 plt.bar 函数绘制条形图
# ages 是 X 轴的位置
# count 是 Y 轴的高度
plt.bar(ages, count)

# --- 关于 plt.bar 参数的注释 ---
# params
# x: 条形图x轴
# y：条形图的高度
# width：条形图的宽度 默认是0.8
# bottom：条形底部的y坐标值 默认是0
# align：center / edge 条形图是否以x轴坐标为中心点或者是以x轴坐标为边缘
# ---

# --- 3. 显示图例 ---
# 尝试显示图例
# (注意：由于 plt.bar 没有设置 label 参数，图例将显示为空)
plt.legend()


# --- 4. 显示图像 ---
# 在窗口中显示生成的图表
plt.show()