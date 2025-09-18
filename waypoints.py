# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import pi
import config

# 定义角度转弧度的转换系数（π/180）
deg2rad = pi/180.0

def makeWaypoints():
    # 设置平均飞行速度为1.6（单位可能是m/s）
    v_average = 8

    # 设置初始时间为3秒（起飞或准备阶段的时间）
    t_ini = 3
    # 定义四个航段的时间间隔数组
    # 值为0表示该航点是一个过渡点（不停留）
    t = np.array([2, 0, 2, 0])
    
    # 设置初始位置坐标为原点(0,0,0) - x,y,z三个维度
    wp_ini = np.array([0, 0, 0])
    # 定义航点坐标数组，包含4个目标点和最后回到初始点
    # 每个航点是三维坐标 [x, y, z]
    wp = np.array([[20, 20, 10],        # 航点1: x=2, y=2, z=1
                   [-20, 30, -30],      # 航点2: x=-2, y=3, z=-3  
                   [-20, -10, -30],     # 航点3: x=-2, y=-1, z=-3
                   [30, -20, 10],       # 航点4: x=3, y=-2, z=1
                   wp_ini])             # 航点5: 回到初始位置

    # 设置初始偏航角为0度（正前方）
    yaw_ini = 0    
    # 定义四个航点处的偏航角（单位为度）
    yaw = np.array([20, -90, 120, 45])

    # 数据处理部分：
    # 将初始时间和时间间隔合并成一个完整的时间序列，并转换为浮点型
    t = np.hstack((t_ini, t)).astype(float)                 # 时间序列
    # 将初始位置和航点坐标垂直堆叠，形成完整的航点序列，转换为浮点型
    wp = np.vstack((wp_ini, wp)).astype(float)              # 航点序列
    # 将初始偏航角和航点偏航角水平堆叠，转换为浮点型，并从角度转换为弧度
    yaw = np.hstack((yaw_ini, yaw)).astype(float)*deg2rad   # 航向序列

    # 返回四个值：时间数组、航点坐标数组、偏航角数组（弧度制）、平均速度
    return t, wp, yaw, v_average

