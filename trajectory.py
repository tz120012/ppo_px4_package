# -*- coding: utf-8 -*-
"""
作者: John Bass
邮箱: john.bobzwik@gmail.com
许可证: MIT
欢迎使用和修改此代码，但请保留以上信息。谢谢！
"""
# 函数 get_poly_cc, minSomethingTraj, pos_waypoint_min 源自 Peter Huang 的工作:
# https://github.com/hbd730/quadcopter-simulation
# 作者: Peter Huang
# 邮箱: hbd730@gmail.com
# 许可证: BSD
# 欢迎使用和修改此代码，但请保留以上信息。谢谢！



import numpy as np
from numpy import pi
from numpy.linalg import norm
from waypoints import makeWaypoints
import config

# class Trajectory:

#     def __init__(self, quad, ctrlType, trajSelect):

#         self.ctrlType = ctrlType                # 位置控制：xyz_pos
#         self.xyzType = trajSelect[0]            # 轨迹类型：minimum jerk，可以通过这个参数定义参考输入
#         self.yawType = trajSelect[1]            # 航向控制: follow
#         self.averVel = trajSelect[2]            # 航点时间：average velocity

#         t_wps, wps, y_wps, v_wp = makeWaypoints()       # 生成航点
#         self.t_wps = t_wps          # 航点时间
#         self.wps   = wps            # 航点
#         self.y_wps = y_wps          # 航向
#         self.v_wp  = v_wp           # 航点速度

#         self.end_reached = 0

#         # 如果控制类型是位置控制（xyz_pos）
#         if (self.ctrlType == "xyz_pos"):
#             # 计算各航段的时间间隔（t_wps是航点时间数组，np.diff计算相邻时间差）
#             self.T_segment = np.diff(self.t_wps)

#             # 如果使用平均速度模式（默认选项）
#             if (self.averVel == 1):         
#                 # 计算相邻航点之间的欧几里得距离（三维空间距离）
#                 distance_segment = self.wps[1:] - self.wps[:-1]
#                 # 重新计算各航段时间：距离除以平均速度
#                 self.T_segment = np.sqrt(distance_segment[:,0]**2 + distance_segment[:,1]**2 + distance_segment[:,2]**2)/self.v_wp
#                 # 重新构建时间数组：第一个航点时间为0，后面是累计时间
#                 self.t_wps = np.zeros(len(self.T_segment) + 1)
#                 self.t_wps[1:] = np.cumsum(self.T_segment)
            
#             # 如果轨迹类型在3-6之间（最小速度、加速度、跃度或冲击轨迹）
#             if (self.xyzType >= 3 and self.xyzType <= 6):
#                 # 计算要最小化的导数阶数：xyzType减去2
#                 # 例如：xyzType=3 -> 最小速度（一阶导数），xyzType=4 -> 最小加速度（二阶导数）
#                 self.deriv_order = int(self.xyzType-2)       

#                 # 为x、y、z三个维度分别计算轨迹系数
#                 # minSomethingTraj函数生成经过所有航点的最小导数轨迹
#                 self.coeff_x = minSomethingTraj(self.wps[:,0], self.T_segment, self.deriv_order)
#                 self.coeff_y = minSomethingTraj(self.wps[:,1], self.T_segment, self.deriv_order)
#                 self.coeff_z = minSomethingTraj(self.wps[:,2], self.T_segment, self.deriv_order)

#             # 如果轨迹类型在7-9之间（在每个航点完全停止的最小轨迹）
#             elif (self.xyzType >= 7 and self.xyzType <= 9):
#                 # 计算要最小化的导数阶数：xyzType减去5
#                 # 例如：xyzType=7 -> 最小加速度（二阶导数），xyzType=8 -> 最小跃度（三阶导数）
#                 self.deriv_order = int(self.xyzType-5)       

#                 # 为x、y、z三个维度分别计算轨迹系数
#                 # minSomethingTraj_stop函数生成在每个航点速度、加速度都为零的轨迹
#                 self.coeff_x = minSomethingTraj_stop(self.wps[:,0], self.T_segment, self.deriv_order)
#                 self.coeff_y = minSomethingTraj_stop(self.wps[:,1], self.T_segment, self.deriv_order)
#                 self.coeff_z = minSomethingTraj_stop(self.wps[:,2], self.T_segment, self.deriv_order)
            
#             # 如果轨迹类型在10-11之间（快速停止轨迹，只在航点速度为零）
#             elif (self.xyzType >= 10 and self.xyzType <= 11):
#                 # 计算要最小化的导数阶数：xyzType减去7
#                 # 例如：xyzType=10 -> 最小加加速度（三阶导数），xyzType=11 -> 最小冲击（四阶导数）
#                 self.deriv_order = int(self.xyzType-7)       

#                 # 为x、y、z三个维度分别计算轨迹系数
#                 # minSomethingTraj_faststop函数生成只在航点速度为零，加速度连续的轨迹
#                 self.coeff_x = minSomethingTraj_faststop(self.wps[:,0], self.T_segment, self.deriv_order)
#                 self.coeff_y = minSomethingTraj_faststop(self.wps[:,1], self.T_segment, self.deriv_order)
#                 self.coeff_z = minSomethingTraj_faststop(self.wps[:,2], self.T_segment, self.deriv_order)

        
#         if (self.yawType == 4):
#             self.y_wps = np.zeros(len(self.t_wps))
        
#         # Get initial heading
#         self.current_heading = quad.psi
        
#         # Initialize trajectory setpoint
#         self.desPos = np.zeros(3)    # 期望位置 (x, y, z)
#         self.desVel = np.zeros(3)    # 期望速度 (xdot, ydot, zdot)
#         self.desAcc = np.zeros(3)    # 期望加速度 (xdotdot, ydotdot, zdotdot)
#         self.desThr = np.zeros(3)    # 期望推力（N-E-D方向或E-N-U方向，取决于选择）
#         self.desEul = np.zeros(3)    # 期望姿态（世界坐标系中的phi, theta, psi）
#         self.desPQR = np.zeros(3)    # 期望角速度（机体坐标系中的p, q, r）
#         self.desYawRate = 0.         # 期望偏航速率
#         self.sDes = np.hstack((self.desPos, self.desVel, self.desAcc, self.desThr, self.desEul, self.desPQR, self.desYawRate)).astype(float)

#     def desiredState(self, t, Ts, quad):
#         """该函数根据当前时间 t 和轨迹类型，计算并返回期望状态 sDes。"""
#         self.desPos = np.zeros(3)    # 期望位置 (x, y, z)
#         self.desVel = np.zeros(3)    # 期望速度 (xdot, ydot, zdot)
#         self.desAcc = np.zeros(3)    # 期望加速度 (xdotdot, ydotdot, zdotdot)
#         self.desThr = np.zeros(3)    # 期望推力（N-E-D方向或E-N-U方向，取决于选择）
#         self.desEul = np.zeros(3)    # 期望姿态（世界坐标系中的phi, theta, psi）
#         self.desPQR = np.zeros(3)    # 期望角速度（机体坐标系中的p, q, r）
#         self.desYawRate = 0.         # 期望偏航速率

#         def pos_waypoint_timed():
#             """按照时间设置航点位置"""
#             if not (len(self.t_wps) == self.wps.shape[0]):          # 检查时间数组和航点数组大小是否一致
#                 raise Exception("时间数组和航点数组大小不一致。")
#             elif (np.diff(self.t_wps) <= 0).any():
#                 raise Exception("时间数组未正确排序。")  
            
#             if (t == 0):            # 初始航点
#                 self.t_idx = 0      # 航点索引=0
#             elif (t >= self.t_wps[-1]):     # 最后一个航点
#                 self.t_idx = -1             # 航点索引=-1
#             else:
#                 self.t_idx = np.where(t <= self.t_wps)[0][0] - 1    # 确定时间t时刻处于哪一个航段时间段内，寻找下左侧参考航点的索引
            
#             self.desPos = self.wps[self.t_idx,:]                    # 左侧参考航点的位置坐标
                            
#         def pos_waypoint_interp():
#             """ 该函数在航点之间线性插值位置，确保在每个航点时间点到达对应位置。 """
#             if not (len(self.t_wps) == self.wps.shape[0]):
#                 raise Exception("时间数组和航点数组大小不一致。")
#             elif (np.diff(self.t_wps) <= 0).any():
#                 raise Exception("时间数组未正确排序。") 

#             if (t == 0):                        # 当前航点为初始航点
#                 self.t_idx = 0
#                 self.desPos = self.wps[0,:]     # 期望航点[0]
#             elif (t >= self.t_wps[-1]):         # 当前航点为最后一个航点
#                 self.t_idx = -1
#                 self.desPos = self.wps[-1,:]    # 期望航点[-1]
#             else:                               # 当前航点为[0]~[-1]之间的航点，用左右两端点位置的线性插值计算期望位置
#                 self.t_idx = np.where(t <= self.t_wps)[0][0] - 1
#                 scale = (t - self.t_wps[self.t_idx])/self.T_segment[self.t_idx]
#                 self.desPos = (1 - scale) * self.wps[self.t_idx,:] + scale * self.wps[self.t_idx + 1,:]
        
#         def pos_waypoint_min():
#             """该函数根据已知航点和时间，生成一个最小速度(vel)、加速度(acc)、跃度(jerk)或冲击(snap)的轨迹，经过每个航点。
#             输出为时间 t 时与下一个航点相关的期望状态。
#             """
#             if not (len(self.t_wps) == self.wps.shape[0]):
#                 raise Exception("时间数组和航点数组大小不一致。")
                
#             # 目标是“最小化第r阶导”。标准做法：每段使用2r−1次多项式。因为2r−1次多项式有2r个系数，所以 nb_coeff = deriv_order * 2 
#             nb_coeff = self.deriv_order*2       # 多项式系数数量（与最小化的导数阶数相关）

#             # t=0，在初始位置悬停
#             if t == 0:
#                 self.t_idx = 0
#                 self.desPos = self.wps[0,:]
#             # 在最后一个航点位置悬停
#             elif (t >= self.t_wps[-1]):
#                 self.t_idx = -1
#                 self.desPos = self.wps[-1,:]
#             else:
#                 self.t_idx = np.where(t <= self.t_wps)[0][0] - 1
                
#                 # 缩放时间（在0和该段持续时间之间）
#                 scale = (t - self.t_wps[self.t_idx])
                
#                 # 所有段的系数被串接存放在一维数组 coeff_x 中，按段顺序拼在一起。对第i段，取出对应的区间：
#                 start = nb_coeff * self.t_idx       # 第t_idx段航迹多项式系数开始索引（左侧边界）
#                 end = nb_coeff * (self.t_idx + 1)   # 第t_idx段航迹多项式系数结束索引（右侧边界）
                
#                 # 设置三轴（x,y,z）的期望位置、速度和加速度
#                 t0 = get_poly_cc(nb_coeff, 0, scale)
#                 self.desPos = np.array([self.coeff_x[start:end].dot(t0), self.coeff_y[start:end].dot(t0), self.coeff_z[start:end].dot(t0)])

#                 t1 = get_poly_cc(nb_coeff, 1, scale)
#                 self.desVel = np.array([self.coeff_x[start:end].dot(t1), self.coeff_y[start:end].dot(t1), self.coeff_z[start:end].dot(t1)])

#                 t2 = get_poly_cc(nb_coeff, 2, scale)
#                 self.desAcc = np.array([self.coeff_x[start:end].dot(t2), self.coeff_y[start:end].dot(t2), self.coeff_z[start:end].dot(t2)])
        
#         def pos_waypoint_arrived():
#             """该函数在到达航点后自动切换到下一个航点。"""
#             dist_consider_arrived = 0.2     # 航点半径（判断到达阈值）到航点的距离小于该值则认为已到达
#             if (t == 0):
#                 self.t_idx = 0
#                 self.end_reached = 0
#             elif not(self.end_reached):
#                 # 当前位置到目标航点的欧氏距离
#                 distance_to_next_wp = ((self.wps[self.t_idx,0]-quad.pos[0])**2 + (self.wps[self.t_idx,1]-quad.pos[1])**2 + (self.wps[self.t_idx,2]-quad.pos[2])**2)**(0.5)
#                 if (distance_to_next_wp < dist_consider_arrived):
#                     self.t_idx += 1
#                     if (self.t_idx >= len(self.wps[:,0])):    # 如果 t_idx 已到达规划航点末尾
#                         self.end_reached = 1
#                         self.t_idx = -1
                    
#             self.desPos = self.wps[self.t_idx,:]    # 航点列表查找对应航点

#         def pos_waypoint_arrived_wait():
#             """该函数在到达航点后自动切换到下一个航点，并在每个航点停留指定时间。"""
#             dist_consider_arrived = 0.2     # 到航点的距离小于该值则认为已到达
#             if (t == 0):
#                 self.t_idx = 0              # 要前往的航点索引
#                 self.t_arrived = 0          # 到达第一个航点（[0, 0, 0]）的时间
#                 self.arrived = True         # 是否已到达第一个航点的布尔值
#                 self.end_reached = 0        # 尚未到达终点
            
            
#             elif not(self.end_reached):     # 如果没有到达终点，计算到达下一个航点的距离
#                 distance_to_next_wp = ((self.wps[self.t_idx,0]-quad.pos[0])**2 + (self.wps[self.t_idx,1]-quad.pos[1])**2 + (self.wps[self.t_idx,2]-quad.pos[2])**2)**(0.5)
                
#                 # 如果距离小于阈值，记录到达时间并确认到达
#                 if (distance_to_next_wp < dist_consider_arrived) and not self.arrived:
#                     self.t_arrived = t
#                     self.arrived = True

#                 # 如果已到达并等待超过指定时间，则航点索引加一
#                 # 可将 'self.t_wps[self.t_idx]' 替换为任意数值，实现所有航点固定等待时间
#                 elif self.arrived and (t-self.t_arrived > self.t_wps[self.t_idx]):   
#                     self.t_idx += 1
#                     self.arrived = False

#                     # 如果 t_idx 已到达规划航点末尾
#                     if (self.t_idx >= len(self.wps[:,0])):    
#                         self.end_reached = 0                  # 设为 1 停止循环
#                         self.t_idx = 0                        # 设为 -1 停止循环  
                    
#             self.desPos = self.wps[self.t_idx,:]

#         def yaw_waypoint_timed():
#             """根据时间点设置偏航角"""
#             # 检查时间数组和偏航数组大小是否一致
#             if not (len(self.t_wps) == len(self.y_wps)):
#                 raise Exception("时间数组和偏航数组大小不一致。")
            
#             self.desEul[2] = self.y_wps[self.t_idx]
                    

#         def yaw_waypoint_interp():
#             """根据时间点线性插值偏航角"""
#             if not (len(self.t_wps) == len(self.y_wps)):
#                 raise Exception("时间数组和偏航数组大小不一致。")

#             if (t == 0) or (t >= self.t_wps[-1]):
#                 self.desEul[2] = self.y_wps[self.t_idx]
#             else:
#                 scale = (t - self.t_wps[self.t_idx])/self.T_segment[self.t_idx]
#                 self.desEul[2] = (1 - scale)*self.y_wps[self.t_idx] + scale*self.y_wps[self.t_idx + 1]
                
#                 # 当前向量与下一个航向向量的夹角
#                 delta_psi = self.desEul[2] - self.current_heading
#                 # 设置偏航速率
#                 self.desYawRate = delta_psi / Ts 
#                 # 准备下一次迭代
#                 self.current_heading = self.desEul[2]
        

#         def yaw_follow():
#             """航向跟随：使无人机的航向与其期望速度方向一致"""
#             if (self.xyzType == 1 or self.xyzType == 2 or self.xyzType == 12):
#                 if (t == 0):
#                     self.desEul[2] = 0
#                 else:
#                     # 计算期望偏航角
#                     self.desEul[2] = np.arctan2(self.desPos[1]-quad.pos[1], self.desPos[0]-quad.pos[0])
            
#             elif (self.xyzType == 13):
#                 if (t == 0):
#                     self.desEul[2] = 0
#                     self.prevDesYaw = self.desEul[2]
#                 else:
#                     if not (self.arrived):
#                         # 计算期望偏航角
#                         self.desEul[2] = np.arctan2(self.desPos[1]-quad.pos[1], self.desPos[0]-quad.pos[0])
#                         self.prevDesYaw = self.desEul[2]
#                     else:
#                         self.desEul[2] = self.prevDesYaw

#             else:
#                 if (t == 0) or (t >= self.t_wps[-1]):           # 初始航点或最后一个航点
#                     self.desEul[2] = self.y_wps[self.t_idx]     # 查找航向列表的值
#                 else:
#                     # 计算期望偏航角
#                     self.desEul[2] = np.arctan2(self.desVel[1], self.desVel[0])
                    
#             # 粗暴处理：检测 desEul[2] 从 -pi 跳到 pi（或反之），手动切换 current_heading 
#             if (np.sign(self.desEul[2]) - np.sign(self.current_heading) and abs(self.desEul[2]-self.current_heading) >= 2*pi-0.1):
#                 self.current_heading = self.current_heading + np.sign(self.desEul[2])*2*pi
            
#             # 当前向量与下一个航向向量的夹角
#             delta_psi = self.desEul[2] - self.current_heading
            
#             # 设置偏航速率
#             self.desYawRate = delta_psi / Ts 

#             # 准备下一次迭代
#             self.current_heading = self.desEul[2]


#         if (self.ctrlType == "xyz_vel"):
#             if (self.xyzType == 1):
#                 self.sDes = testVelControl(t)

#         elif (self.ctrlType == "xy_vel_z_pos"):
#             if (self.xyzType == 1):
#                 self.sDes = testVelControl(t)
        
#         elif (self.ctrlType == "xyz_pos"):
#             # Hover at [0, 0, 0]
#             if (self.xyzType == 0):
#                 pass 
#             # For simple testing
#             elif (self.xyzType == 99):
#                 self.sDes = testXYZposition(t)   
#             else:    
#                 # 位置轨迹类型列表
#                 # ---------------------------
#                 # 在每个 t_wps[i] 设置期望位置
#                 if (self.xyzType == 1):
#                     pos_waypoint_timed()
#                 # 在每个航点之间插值，确保在每个 t_wps[i] 到达期望位置
#                 elif (self.xyzType == 2):
#                     pos_waypoint_interp()
#                 # 计算最小速度、加速度、跃度或冲击轨迹
#                 elif (self.xyzType >= 3 and self.xyzType <= 11):
#                     pos_waypoint_min()
#                 # 到达航点后切换到下一个航点
#                 elif (self.xyzType == 12):
#                     pos_waypoint_arrived()
#                 # 到达航点后等待指定时间再切换到下一个航点
#                 elif (self.xyzType == 13):
#                     pos_waypoint_arrived_wait()
                
#                 # 航向轨迹类型列表
#                 # ---------------------------
#                 # 在每个 t_wps[i] 设置期望航向角
#                 if (self.yawType == 0):
#                     pass
#                 elif (self.yawType == 1):
#                     yaw_waypoint_timed()
#                 # 在每个航点之间插值，确保在每个 t_wps[i] 到达期望航向角
#                 elif (self.yawType == 2):
#                     yaw_waypoint_interp()
#                 # 使无人机航向与期望速度方向一致
#                 elif (self.yawType == 3):       # 默认选项
#                     yaw_follow()

#                 self.sDes = np.hstack((self.desPos, self.desVel, self.desAcc, self.desThr, self.desEul, self.desPQR, self.desYawRate)).astype(float)
        
#         return self.sDes

class Trajectory:

    def __init__(self, quad, ctrlType, trajSelect):

        self.ctrlType = ctrlType                # 位置控制：xyz_pos
        self.xyzType = trajSelect[0]            # 轨迹类型：minimum jerk，可以通过这个参数定义参考输入
        self.yawType = trajSelect[1]            # 航向控制: follow
        self.averVel = trajSelect[2]            # 航点时间：average velocity

        t_wps, wps, y_wps, v_wp = makeWaypoints()       # 生成航点
        self.t_wps = t_wps          # 航点时间
        self.wps   = wps            # 航点
        self.y_wps = y_wps          # 航向
        self.v_wp  = v_wp           # 航点速度

        self.end_reached = 0

        # 如果控制类型是位置控制（xyz_pos）
        if (self.ctrlType == "xyz_pos"):
            # 计算各航段的时间间隔（t_wps是航点时间数组，np.diff计算相邻时间差）
            self.T_segment = np.diff(self.t_wps)

            # 如果使用平均速度模式（默认选项）
            if (self.averVel == 1):         
                # 计算相邻航点之间的欧几里得距离（三维空间距离）
                distance_segment = self.wps[1:] - self.wps[:-1]
                # 重新计算各航段时间：距离除以平均速度
                self.T_segment = np.sqrt(distance_segment[:,0]**2 + distance_segment[:,1]**2 + distance_segment[:,2]**2)/self.v_wp
                # 重新构建时间数组：第一个航点时间为0，后面是累计时间
                self.t_wps = np.zeros(len(self.T_segment) + 1)
                self.t_wps[1:] = np.cumsum(self.T_segment)
            
            # 如果轨迹类型在3-6之间（最小速度、加速度、跃度或冲击轨迹）
            if (self.xyzType >= 3 and self.xyzType <= 6):
                # 计算要最小化的导数阶数：xyzType减去2
                # 例如：xyzType=3 -> 最小速度（一阶导数），xyzType=4 -> 最小加速度（二阶导数）
                self.deriv_order = int(self.xyzType-2)       

                # 为x、y、z三个维度分别计算轨迹系数
                # minSomethingTraj函数生成经过所有航点的最小导数轨迹
                self.coeff_x = minSomethingTraj(self.wps[:,0], self.T_segment, self.deriv_order)
                self.coeff_y = minSomethingTraj(self.wps[:,1], self.T_segment, self.deriv_order)
                self.coeff_z = minSomethingTraj(self.wps[:,2], self.T_segment, self.deriv_order)

            # 如果轨迹类型在7-9之间（在每个航点完全停止的最小轨迹）
            elif (self.xyzType >= 7 and self.xyzType <= 9):
                # 计算要最小化的导数阶数：xyzType减去5
                # 例如：xyzType=7 -> 最小加速度（二阶导数），xyzType=8 -> 最小跃度（三阶导数）
                self.deriv_order = int(self.xyzType-5)       

                # 为x、y、z三个维度分别计算轨迹系数
                # minSomethingTraj_stop函数生成在每个航点速度、加速度都为零的轨迹
                self.coeff_x = minSomethingTraj_stop(self.wps[:,0], self.T_segment, self.deriv_order)
                self.coeff_y = minSomethingTraj_stop(self.wps[:,1], self.T_segment, self.deriv_order)
                self.coeff_z = minSomethingTraj_stop(self.wps[:,2], self.T_segment, self.deriv_order)
            
            # 如果轨迹类型在10-11之间（快速停止轨迹，只在航点速度为零）
            elif (self.xyzType >= 10 and self.xyzType <= 11):
                # 计算要最小化的导数阶数：xyzType减去7
                # 例如：xyzType=10 -> 最小加加速度（三阶导数），xyzType=11 -> 最小冲击（四阶导数）
                self.deriv_order = int(self.xyzType-7)       

                # 为x、y、z三个维度分别计算轨迹系数
                # minSomethingTraj_faststop函数生成只在航点速度为零，加速度连续的轨迹
                self.coeff_x = minSomethingTraj_faststop(self.wps[:,0], self.T_segment, self.deriv_order)
                self.coeff_y = minSomethingTraj_faststop(self.wps[:,1], self.T_segment, self.deriv_order)
                self.coeff_z = minSomethingTraj_faststop(self.wps[:,2], self.T_segment, self.deriv_order)

        
        if (self.yawType == 4):
            self.y_wps = np.zeros(len(self.t_wps))
        
        # Get initial heading
        self.current_heading = quad.psi
        
        # Initialize trajectory setpoint
        self.desPos = np.zeros(3)    # 期望位置 (x, y, z)
        self.desVel = np.zeros(3)    # 期望速度 (xdot, ydot, zdot)
        self.desAcc = np.zeros(3)    # 期望加速度 (xdotdot, ydotdot, zdotdot)
        self.desThr = np.zeros(3)    # 期望推力（N-E-D方向或E-N-U方向，取决于选择）
        self.desEul = np.zeros(3)    # 期望姿态（世界坐标系中的phi, theta, psi）
        self.desPQR = np.zeros(3)    # 期望角速度（机体坐标系中的p, q, r）
        self.desYawRate = 0.         # 期望偏航速率
        self.sDes = np.hstack((self.desPos, self.desVel, self.desAcc, self.desThr, self.desEul, self.desPQR, self.desYawRate)).astype(float)

    def desiredState(self, t, Ts, quad):
        """该函数根据当前时间 t 和轨迹类型，计算并返回期望状态 sDes。"""
        self.desPos = np.zeros(3)    # 期望位置 (x, y, z)
        self.desVel = np.zeros(3)    # 期望速度 (xdot, ydot, zdot)
        self.desAcc = np.zeros(3)    # 期望加速度 (xdotdot, ydotdot, zdotdot)
        self.desThr = np.zeros(3)    # 期望推力（N-E-D方向或E-N-U方向，取决于选择）
        self.desEul = np.zeros(3)    # 期望姿态（世界坐标系中的phi, theta, psi）
        self.desPQR = np.zeros(3)    # 期望角速度（机体坐标系中的p, q, r）
        self.desYawRate = 0.         # 期望偏航速率

        # 轨迹循环：针对按时间驱动的轨迹类型，将 t 周期化为 t_mod
        use_time_driven = (self.ctrlType == "xyz_pos") and (
            self.xyzType in [1, 2] or (3 <= self.xyzType <= 11)
        )
        if use_time_driven:
            T_total = float(self.t_wps[-1] - self.t_wps[0]) if len(self.t_wps) > 1 else 0.0
            if T_total <= 0:
                t_mod = t
            else:
                t_shift = t - self.t_wps[0]
                t_mod_shift = np.mod(t_shift, T_total)  # 支持 t<0
                t_mod = self.t_wps[0] + t_mod_shift
                # 避免恰落在周期末端，导致索引抖动
                eps = 1e-9
                if abs(t_mod - self.t_wps[-1]) < eps:
                    t_mod = self.t_wps[-1] - eps
        else:
            t_mod = t

        def pos_waypoint_timed():
            """按照时间设置航点位置（循环）。"""
            if not (len(self.t_wps) == self.wps.shape[0]):          # 检查时间数组和航点数组大小是否一致
                raise Exception("时间数组和航点数组大小不一致。")
            elif (np.diff(self.t_wps) <= 0).any():
                raise Exception("时间数组未正确排序。")  
            
            # 使用 t_mod 进行段定位
            if t_mod <= self.t_wps[0]:
                self.t_idx = 0
            elif t_mod >= self.t_wps[-1]:
                self.t_idx = -1
            else:
                self.t_idx = np.where(t_mod <= self.t_wps)[0][0] - 1
            
            self.desPos = self.wps[self.t_idx,:]                    # 左侧参考航点的位置坐标
                            
        def pos_waypoint_interp():
            """在航点之间线性插值位置（循环），确保在每个航点时间点到达对应位置。"""
            if not (len(self.t_wps) == self.wps.shape[0]):
                raise Exception("时间数组和航点数组大小不一致。")
            elif (np.diff(self.t_wps) <= 0).any():
                raise Exception("时间数组未正确排序。") 

            if t_mod <= self.t_wps[0]:
                self.t_idx = 0
                self.desPos = self.wps[0,:]
            elif t_mod >= self.t_wps[-1]:
                self.t_idx = -1
                self.desPos = self.wps[-1,:]
            else:
                self.t_idx = np.where(t_mod <= self.t_wps)[0][0] - 1
                scale = (t_mod - self.t_wps[self.t_idx])/self.T_segment[self.t_idx]
                self.desPos = (1 - scale) * self.wps[self.t_idx,:] + scale * self.wps[self.t_idx + 1,:]
        
        def pos_waypoint_min():
            """最小速度/加速度/跃度/冲击轨迹（循环），输出与当前段相关的期望状态。"""
            if not (len(self.t_wps) == self.wps.shape[0]):
                raise Exception("时间数组和航点数组大小不一致。")
                
            # 目标是“最小化第r阶导”。每段使用 2r−1 次多项式，系数数为 2r
            nb_coeff = self.deriv_order*2

            if t_mod <= self.t_wps[0]:
                self.t_idx = 0
                self.desPos = self.wps[0,:]
            elif t_mod >= self.t_wps[-1]:
                self.t_idx = -1
                self.desPos = self.wps[-1,:]
            else:
                self.t_idx = np.where(t_mod <= self.t_wps)[0][0] - 1
                # 段内时间
                scale = (t_mod - self.t_wps[self.t_idx])
                
                # 取出该段多项式系数
                start = nb_coeff * self.t_idx
                end = nb_coeff * (self.t_idx + 1)
                
                # 三轴（x,y,z）的期望位置、速度和加速度
                t0 = get_poly_cc(nb_coeff, 0, scale)
                self.desPos = np.array([
                    self.coeff_x[start:end].dot(t0),
                    self.coeff_y[start:end].dot(t0),
                    self.coeff_z[start:end].dot(t0)
                ])

                t1 = get_poly_cc(nb_coeff, 1, scale)
                self.desVel = np.array([
                    self.coeff_x[start:end].dot(t1),
                    self.coeff_y[start:end].dot(t1),
                    self.coeff_z[start:end].dot(t1)
                ])

                t2 = get_poly_cc(nb_coeff, 2, scale)
                self.desAcc = np.array([
                    self.coeff_x[start:end].dot(t2),
                    self.coeff_y[start:end].dot(t2),
                    self.coeff_z[start:end].dot(t2)
                ])
        
        def pos_waypoint_arrived():
            """到达触发：到达航点后切换到下一个航点（循环回起点）。"""
            dist_consider_arrived = 0.2     # 航点半径（判断到达阈值）
            if (t == 0):
                self.t_idx = 0
                self.end_reached = 0
            else:
                distance_to_next_wp = np.linalg.norm(self.wps[self.t_idx,:] - quad.pos)
                if (distance_to_next_wp < dist_consider_arrived):
                    self.t_idx += 1
                    if (self.t_idx >= len(self.wps[:,0])):    # 末尾则循环回起点
                        self.t_idx = 0
                        self.end_reached = 0
            self.desPos = self.wps[self.t_idx,:]    # 航点列表查找对应航点

        def pos_waypoint_arrived_wait():
            """到达触发并等待：到达航点后停留指定时间，再切换到下一个航点（循环回起点）。"""
            dist_consider_arrived = 0.2
            if (t == 0):
                self.t_idx = 0              # 要前往的航点索引
                self.t_arrived = 0          # 到达当前航点的时间
                self.arrived = True         # 是否已到达当前航点
                self.end_reached = 0        # 尚未到达终点
            else:
                distance_to_next_wp = np.linalg.norm(self.wps[self.t_idx,:] - quad.pos)
                # 到达且尚未标记 arrived，则记录时间并标记
                if (distance_to_next_wp < dist_consider_arrived) and not self.arrived:
                    self.t_arrived = t
                    self.arrived = True
                # 若已到达且等待超时，则推进索引
                elif self.arrived and (t - self.t_arrived > self.t_wps[self.t_idx]):
                    self.t_idx += 1
                    self.arrived = False
                    # 循环回起点
                    if (self.t_idx >= len(self.wps[:,0])):    
                        self.t_idx = 0
                        self.end_reached = 0
            self.desPos = self.wps[self.t_idx,:]

        def yaw_waypoint_timed():
            """根据时间点设置偏航角（共享 self.t_idx）。"""
            if not (len(self.t_wps) == len(self.y_wps)):
                raise Exception("时间数组和偏航数组大小不一致。")
            self.desEul[2] = self.y_wps[self.t_idx]
                    

        def yaw_waypoint_interp():
            """根据时间线性插值偏航角（循环，对应位置段索引与 scale）。"""
            if not (len(self.t_wps) == len(self.y_wps)):
                raise Exception("时间数组和偏航数组大小不一致。")

            if (t_mod <= self.t_wps[0]) or (t_mod >= self.t_wps[-1]):
                self.desEul[2] = self.y_wps[self.t_idx]
            else:
                scale = (t_mod - self.t_wps[self.t_idx])/self.T_segment[self.t_idx]
                self.desEul[2] = (1 - scale)*self.y_wps[self.t_idx] + scale*self.y_wps[self.t_idx + 1]
                
                # 当前向量与下一个航向向量的夹角
                delta_psi = self.desEul[2] - self.current_heading
                # 设置偏航速率
                self.desYawRate = delta_psi / Ts 
                # 准备下一次迭代
                self.current_heading = self.desEul[2]
        

        def yaw_follow():
            """航向跟随：使无人机的航向与其期望速度方向一致"""
            if (self.xyzType == 1 or self.xyzType == 2 or self.xyzType == 12):
                if (t == 0):
                    self.desEul[2] = 0
                else:
                    # 计算期望偏航角（基于位置向量）
                    self.desEul[2] = np.arctan2(self.desPos[1]-quad.pos[1], self.desPos[0]-quad.pos[0])
            
            elif (self.xyzType == 13):
                if (t == 0):
                    self.desEul[2] = 0
                    self.prevDesYaw = self.desEul[2]
                else:
                    if not (hasattr(self, "arrived") and self.arrived):
                        # 计算期望偏航角
                        self.desEul[2] = np.arctan2(self.desPos[1]-quad.pos[1], self.desPos[0]-quad.pos[0])
                        self.prevDesYaw = self.desEul[2]
                    else:
                        self.desEul[2] = self.prevDesYaw

            else:
                if (t_mod <= self.t_wps[0]) or (t_mod >= self.t_wps[-1]):           # 初始航点或最后一个航点
                    self.desEul[2] = self.y_wps[self.t_idx]     # 查找航向列表的值
                else:
                    # 计算期望偏航角（基于期望速度）
                    self.desEul[2] = np.arctan2(self.desVel[1], self.desVel[0])
                    
            # 粗暴处理：检测 desEul[2] 从 -pi 跳到 pi（或反之），手动切换 current_heading 
            if (np.sign(self.desEul[2]) - np.sign(self.current_heading) and abs(self.desEul[2]-self.current_heading) >= 2*pi-0.1):
                self.current_heading = self.current_heading + np.sign(self.desEul[2])*2*pi
            
            # 当前向量与下一个航向向量的夹角
            delta_psi = self.desEul[2] - self.current_heading
            
            # 设置偏航速率
            self.desYawRate = delta_psi / Ts 

            # 准备下一次迭代
            self.current_heading = self.desEul[2]


        if (self.ctrlType == "xyz_vel"):
            if (self.xyzType == 1):
                self.sDes = testVelControl(t)

        elif (self.ctrlType == "xy_vel_z_pos"):
            if (self.xyzType == 1):
                self.sDes = testVelControl(t)
        
        elif (self.ctrlType == "xyz_pos"):
            # Hover at [0, 0, 0]
            if (self.xyzType == 0):
                pass 
            # For simple testing
            elif (self.xyzType == 99):
                self.sDes = testXYZposition(t)   
            else:    
                # 位置轨迹类型列表
                # ---------------------------
                # 在每个 t_wps[i] 设置期望位置
                if (self.xyzType == 1):
                    pos_waypoint_timed()
                # 在每个航点之间插值，确保在每个 t_wps[i] 到达期望位置
                elif (self.xyzType == 2):
                    pos_waypoint_interp()
                # 计算最小速度、加速度、跃度或冲击轨迹
                elif (self.xyzType >= 3 and self.xyzType <= 11):
                    pos_waypoint_min()
                # 到达航点后切换到下一个航点（循环）
                elif (self.xyzType == 12):
                    pos_waypoint_arrived()
                # 到达航点后等待指定时间再切换到下一个航点（循环）
                elif (self.xyzType == 13):
                    pos_waypoint_arrived_wait()
                
                # 航向轨迹类型列表
                # ---------------------------
                # 在每个 t_wps[i] 设置期望航向角
                if (self.yawType == 0):
                    pass
                elif (self.yawType == 1):
                    yaw_waypoint_timed()
                # 在每个航点之间插值，确保在每个 t_wps[i] 到达期望航向角
                elif (self.yawType == 2):
                    yaw_waypoint_interp()
                # 使无人机航向与期望速度方向一致
                elif (self.yawType == 3):       # 默认选项
                    yaw_follow()

                self.sDes = np.hstack((self.desPos, self.desVel, self.desAcc, self.desThr, self.desEul, self.desPQR, self.desYawRate)).astype(float)
        
        return self.sDes

def get_poly_cc(n, k, t):
    """这是一个辅助函数，用于获取 n 阶多项式在 t 时刻的 k 阶导数的系数。"""
    assert (n > 0 and k >= 0), "order and derivative must be positive."

    cc = np.ones(n)
    D  = np.linspace(n-1, 0, n)

    for i in range(n):
        for j in range(k):
            cc[i] = cc[i] * D[i]
            D[i] = D[i] - 1
            if D[i] == -1:
                D[i] = 0

    for i, c in enumerate(cc):
        cc[i] = c * np.power(t, D[i])

    return cc


def minSomethingTraj(waypoints, times, order):
    """该函数接收期望航点列表 [x0, x1, x2...xN] 和时间，返回 N+1 个航点（N 段）的 [M*N,1] 系数矩阵，
    其中每段的系数数 M = (order)*2。如果需要最小速度，order=1；最小冲击则 order=4。

    1.问题描述
    生成跨越 N+1 个航点的完整轨迹，由 N 个多项式线段组成。
    每段为 (2*order-1) 阶多项式：
    最小速度：     Pi = ai_0 + ai1*t
    最小加速度：   Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3
    最小跃度：     Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5
    最小冲击：     Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5 + ai6*t^6 + ai7*t^7

    每个多项式有 M 个未知系数，总共需要解 M*N 个未知数，因此需要 M*N 个约束。

    2.约束条件
    通常约束是一组条件，定义初始和最终状态，以及各分段函数之间的连续性，包括在中间航点处轨迹高阶导数的连续性。

    3.矩阵设计
    有 M*N 个未知系数和 M*N 个方程（约束），问题转化为线性方程组。

    A * Coeff = B

    先看 B 矩阵，B 很简单，就是方程右侧的一些常数。共有 M*N 个约束，B 为 [M*N, 1]。
    Coeff 是最终输出的 M*N 个元素的系数矩阵。
    A 矩阵是系数-系数矩阵。我们不再只看某个多项式 Pi，而是整体 P1, P2...PN。
    所以 A 的形状为 [M*N, M*N]。

    每行的每个元素表示在某个约束下 ai_j 的系数。
    """
    n = len(waypoints) - 1
    nb_coeff = order*2

    # 初始化 A 和 B 矩阵
    A = np.zeros([nb_coeff*n, nb_coeff*n])
    B = np.zeros(nb_coeff*n)

    # 填充 B 矩阵
    for i in range(n):
        B[i] = waypoints[i]
        B[i + n] = waypoints[i+1]

    # 约束1 - 每段起始位置
    for i in range(n):
        A[i][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, 0)

    # 约束2 - 每段结束位置
    for i in range(n):
        A[i+n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, times[i])

    # 约束3 - 每段起始导数（到 order 阶）为零
    for k in range(1, order):
        A[2*n+k-1][:nb_coeff] = get_poly_cc(nb_coeff, k, 0)

    # 约束4 - 每段结束导数（到 order 阶）为零
    for k in range(1, order):
        A[2*n+(order-1)+k-1][-nb_coeff:] = get_poly_cc(nb_coeff, k, times[i])
    
    # 约束5 - 所有导数在航点处连续
    for i in range(n-1):
        for k in range(1, nb_coeff-1):
            A[2*n+2*(order-1) + i*2*(order-1)+k-1][i*nb_coeff : (i*nb_coeff+nb_coeff*2)] = np.concatenate((get_poly_cc(nb_coeff, k, times[i]), -get_poly_cc(nb_coeff, k, 0)))
    
    # 求解系数
    Coeff = np.linalg.solve(A, B)
    return Coeff


# 最小加速度/加加速度/加加加速度轨迹，但在每个航点处速度和加速度为零
def minSomethingTraj_stop(waypoints, times, order):
    """该函数接收期望航点列表 [x0, x1, x2...xN] 和时间，返回 N+1 个航点（N 段）的 [M*N,1] 系数矩阵，
    其中每段的系数数 M = (order)*2。如果需要最小加速度，order=2；最小冲击则 order=4。

    1.问题描述
    生成跨越 N+1 个航点的完整轨迹，由 N 个多项式线段组成。
    每段为 (2*order-1) 阶多项式：
    最小速度：     Pi = ai_0 + ai1*t
    最小加速度：   Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3
    最小跃度：     Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5
    最小冲击：     Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5 + ai6*t^6 + ai7*t^7

    每个多项式有 M 个未知系数，总共需要解 M*N 个未知数，因此需要 M*N 个约束。

    与 minSomethingTraj 不同，该函数生成在每个航点速度、加速度和跃度都为零的轨迹。
    这样无人机会在每个航点瞬间停下。
    """

    n = len(waypoints) - 1
    nb_coeff = order*2

    # 初始化 A 和 B 矩阵
    A = np.zeros([nb_coeff*n, nb_coeff*n])
    B = np.zeros(nb_coeff*n)

    # 填充 B 矩阵
    for i in range(n):
        B[i] = waypoints[i]
        B[i + n] = waypoints[i+1]

    # 约束1 - 每段起始位置
    for i in range(n):
        A[i][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, 0)

    # 约束2 - 每段结束位置
    for i in range(n):
        A[i+n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, times[i])

    # 约束3 - 每段起始导数（到 order 阶）为零
    for i in range(n):
        for k in range(1, order):
            A[2*n + k-1 + i*(order-1)][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, k, 0)

    # 约束4 - 每段结束导数（到 order 阶）为零
    for i in range(n):
        for k in range(1, order):
            A[2*n+(order-1)*n + k-1 + i*(order-1)][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, k, times[i])
    
    # 求解系数
    Coeff = np.linalg.solve(A, B)
    return Coeff

# 最小加速度/加加速度/加加加速度轨迹，但在每个航点处仅速度为零
def minSomethingTraj_faststop(waypoints, times, order):
    """该函数接收期望航点列表 [x0, x1, x2...xN] 和时间，返回 N+1 个航点（N 段）的 [M*N,1] 系数矩阵，
    其中每段的系数数 M = (order)*2。如果需要最小加速度，order=2；最小冲击则 order=4。

    1.问题描述
    生成跨越 N+1 个航点的完整轨迹，由 N 个多项式线段组成。
    每段为 (2*order-1) 阶多项式：
    最小速度：     Pi = ai_0 + ai1*t
    最小加速度：   Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3
    最小跃度：     Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5
    最小冲击：     Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5 + ai6*t^6 + ai7*t^7

    每个多项式有 M 个未知系数，总共需要解 M*N 个未知数，因此需要 M*N 个约束。

    与 minSomethingTraj 不同，该函数只在每个航点速度为零，加速度及更高阶导数连续。
    这样无人机会在每个航点瞬间停下，然后沿来向方向离开。
    """

    n = len(waypoints) - 1
    nb_coeff = order*2

    # 初始化 A 和 B 矩阵
    A = np.zeros([nb_coeff*n, nb_coeff*n])
    B = np.zeros(nb_coeff*n)

    # 填充 B 矩阵
    for i in range(n):
        B[i] = waypoints[i]
        B[i + n] = waypoints[i+1]

    # 约束1 - 每段起始位置
    for i in range(n):
        # print(i)
        A[i][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, 0)

    # 约束2 - 每段结束位置
    for i in range(n):
        # print(i+n)
        A[i+n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, times[i])

    # 约束3 - 每段起始速度为零
    for i in range(n):
        # print(i+2*n)
        A[i+2*n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 1, 0)

    # 约束4 - 每段结束速度为零
    for i in range(n):
        # print(i+3*n)
        A[i+3*n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 1, times[i])

    # 约束5 - 每段起始高阶导数（大于速度且到 order 阶）为零
    for k in range(2, order):
        # print(4*n + k-2)
        A[4*n+k-2][:nb_coeff] = get_poly_cc(nb_coeff, k, 0)

    # 约束6 - 每段结束高阶导数（大于速度且到 order 阶）为零
    for k in range(2, order):
        # print(4*n+(order-2) + k-2)
        A[4*n+k-2+(order-2)][-nb_coeff:] = get_poly_cc(nb_coeff, k, times[i])

    # 约束7 - 所有高阶导数在航点处连续
    for i in range(n-1):
        for k in range(2, nb_coeff-2):
            # print(4*n+2*(order-2)+k-2+i*(nb_coeff-4))
            A[4*n+2*(order-2)+k-2+i*(nb_coeff-4)][i*nb_coeff : (i*nb_coeff+nb_coeff*2)] = np.concatenate((get_poly_cc(nb_coeff, k, times[i]), -get_poly_cc(nb_coeff, k, 0)))
            

    # 求解系数
    Coeff = np.linalg.solve(A, B)
    return Coeff



## 测试脚本

def testXYZposition(t):
    desPos = np.array([0., 0., 0.])
    desVel = np.array([0., 0., 0.])
    desAcc = np.array([0., 0., 0.])
    desThr = np.array([0., 0., 0.]) # 期望推力
    desEul = np.array([0., 0., 0.]) # 期望姿态
    desPQR = np.array([0., 0., 0.]) # 期望角速度
    desYawRate = 30.0*pi/180        # 期望偏航速率
    
    if t >= 1 and t < 4:
        desPos = np.array([2, 2, 1])
    elif t >= 4:
        desPos = np.array([2, -2, -2])
        desEul = np.array([0, 0, pi/3])
    
    sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)

    return sDes


def testVelControl(t):
    desPos = np.array([0., 0., 0.])
    desVel = np.array([0., 0., 0.])
    desAcc = np.array([0., 0., 0.])
    desThr = np.array([0., 0., 0.]) # 期望推力
    desEul = np.array([0., 0., 0.]) # 期望姿态
    desPQR = np.array([0., 0., 0.]) # 期望角速度
    desYawRate = 0.                 # 期望偏航速率

    if t >= 1 and t < 4:
        desVel = np.array([3, 2, 0])
    elif t >= 4:
        desVel = np.array([3, -1, 0])
     
    sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)
    
    return sDes



    # Select Position Trajectory Type (
    # 0: hover：悬停                   
    # 1: pos_waypoint_timed     ：按照时间设置航点位置     
    # 2: pos_waypoint_interp    ：该函数在航点之间线性插值位置，确保在每个航点时间点到达对应位置
    # 3: minimum velocity       ：最小速度航迹跟踪
    # 4: minimum accel          ：最小加速度航迹跟踪
    # 5: minimum jerk           ：最小跃度航迹跟踪
    # 6: minimum snap           ：最小冲击航迹跟踪
    # 7: minimum accel_stop     ：最小加速度停止航迹跟踪
    # 8: minimum jerk_stop      ：最小跃度停止航迹跟踪
    # 9: minimum snap_stop      ：最小冲击停止航迹跟踪
    # 10: minimum jerk_full_stop    ：最小跃度完全停止航迹跟踪
    # 11: minimum snap_full_stop    ：最小冲击完全停止航迹跟踪
    # 12: pos_waypoint_arrived      ：该函数在到达航点后自动切换到下一个航点
    # 13: pos_waypoint_arrived_wait ：该函数在到达航点后自动切换到下一个航点，并在每个航点停留指定时间