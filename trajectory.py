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

# Select Position Trajectory Type (
#     0: hover：悬停
#     1: pos_waypoint_timed
#     2: pos_waypoint_interp
#     3: minimum velocity
#     4: minimum accel
#     5: minimum jerk
#     6: minimum snap
#     7: minimum accel_stop
#     8: minimum jerk_stop
#     9: minimum snap_stop
#     10: minimum jerk_full_stop
#     11: minimum snap_full_stop
#     12: pos_waypoint_arrived
#     13: pos_waypoint_arrived_wait
# )

# -----------------------------
# 多项式与轨迹构建工具
# -----------------------------

def get_poly_cc(n, k, t):
    """获取 n 阶多项式在 t 时刻的 k 阶导数的系数。"""
    assert (n > 0 and k >= 0), "order and derivative must be positive."
    cc = np.ones(n)
    D  = np.linspace(n-1, 0, n)
    for i in range(n):
        for _ in range(k):
            cc[i] = cc[i] * D[i]
            D[i] = D[i] - 1
            if D[i] == -1:
                D[i] = 0
    for i, c in enumerate(cc):
        cc[i] = c * np.power(t, D[i])
    return cc

# -----------------------------
# 扫频/扫幅工具（Ping-Pong 往返扫）
# -----------------------------

def triangle_phase(t, T):
    """
    将时间映射到 [0,1] 的三角波相位：
    t:   0     T/2     T     3T/2   2T ...
    phi: 0 -> 1 -> 0 -> 1 -> 0  ...
    """
    if T <= 0:
        return 1.0
    r = (t % T) / T
    return 2*r if r < 0.5 else 2*(1 - r)

def sine_sweep_pingpong(t, low=0.2, high=2.0, T=20.0, mode="lin"):
    """
    频率往返扫（Hz）：在 [0, T] 内 low->high->low ...
    - mode: "lin" 线性, "log" 对数
    """
    if T <= 0:
        return high
    phi = triangle_phase(t, T)
    if mode == "log":
        low = max(low, 1e-6)
        return low * (high / low) ** phi
    else:
        return low + (high - low) * phi

def amp_sweep_pingpong(t, low=0.1, high=1.0, T=20.0, mode="lin"):
    """
    幅值往返扫：在 [0, T] 内 low->high->low ...
    - mode: "lin" 线性, "log" 对数（对幅值意义弱，一般用线性）
    """
    if T <= 0:
        return high
    phi = triangle_phase(t, T)
    if mode == "log":
        low = max(low, 1e-6)
        return low * (high / low) ** phi
    else:
        return low + (high - low) * phi

def bounded(value, lo, hi):
    return max(lo, min(hi, value))

# -----------------------------
# 最小导数类轨迹构建
# -----------------------------

def minSomethingTraj(waypoints, times, order):
    """
    N 段多项式，阶数为 (2*order-1)，构建最小某阶导数的轨迹。
    返回 Coeff: [M*N], M = 2*order
    """
    n = len(waypoints) - 1
    nb_coeff = order*2
    A = np.zeros([nb_coeff*n, nb_coeff*n])
    B = np.zeros(nb_coeff*n)

    for i in range(n):
        B[i] = waypoints[i]
        B[i + n] = waypoints[i+1]

    # 起点/终点位置
    for i in range(n):
        A[i][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, 0)
        A[i+n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, times[i])

    # 起点导数(1..order-1) = 0
    for k in range(1, order):
        A[2*n+k-1][:nb_coeff] = get_poly_cc(nb_coeff, k, 0)

    # 终点导数(1..order-1) = 0
    for k in range(1, order):
        A[2*n+(order-1)+k-1][-nb_coeff:] = get_poly_cc(nb_coeff, k, times[-1])

    # 中间航点连续性
    for i in range(n-1):
        for k in range(1, nb_coeff-1):
            A[2*n+2*(order-1) + i*2*(order-1)+k-1][i*nb_coeff:(i*nb_coeff+nb_coeff*2)] = np.concatenate(
                (get_poly_cc(nb_coeff, k, times[i]), -get_poly_cc(nb_coeff, k, 0))
            )

    Coeff = np.linalg.solve(A, B)
    return Coeff

def minSomethingTraj_stop(waypoints, times, order):
    """
    在每个航点速度、加速度...为零（停住）的最小导数轨迹。
    """
    n = len(waypoints) - 1
    nb_coeff = order*2
    A = np.zeros([nb_coeff*n, nb_coeff*n])
    B = np.zeros(nb_coeff*n)

    for i in range(n):
        B[i] = waypoints[i]
        B[i + n] = waypoints[i+1]

    for i in range(n):
        A[i][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, 0)
        A[i+n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, times[i])

    # 每段起止导数(1..order-1)=0
    for i in range(n):
        for k in range(1, order):
            A[2*n + k-1 + i*(order-1)][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, k, 0)
    for i in range(n):
        for k in range(1, order):
            A[2*n+(order-1)*n + k-1 + i*(order-1)][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, k, times[i])

    Coeff = np.linalg.solve(A, B)
    return Coeff

def minSomethingTraj_faststop(waypoints, times, order):
    """
    在每个航点速度为零（加速度连续）的最小导数轨迹。
    """
    n = len(waypoints) - 1
    nb_coeff = order*2
    A = np.zeros([nb_coeff*n, nb_coeff*n])
    B = np.zeros(nb_coeff*n)

    for i in range(n):
        B[i] = waypoints[i]
        B[i + n] = waypoints[i+1]

    for i in range(n):
        A[i][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, 0)
        A[i+n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, times[i])

    # 起止速度=0
    for i in range(n):
        A[i+2*n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 1, 0)
    for i in range(n):
        A[i+3*n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 1, times[i])

    # 起止高阶导数（>1 .. order-1）=0
    for k in range(2, order):
        A[4*n+k-2][:nb_coeff] = get_poly_cc(nb_coeff, k, 0)
    for k in range(2, order):
        A[4*n+k-2+(order-2)][-nb_coeff:] = get_poly_cc(nb_coeff, k, times[-1])

    # 中间航点高阶导连续
    for i in range(n-1):
        for k in range(2, nb_coeff-2):
            A[4*n+2*(order-2)+k-2+i*(nb_coeff-4)][i*nb_coeff:(i*nb_coeff+nb_coeff*2)] = np.concatenate(
                (get_poly_cc(nb_coeff, k, times[i]), -get_poly_cc(nb_coeff, k, 0))
            )

    Coeff = np.linalg.solve(A, B)
    return Coeff

# -----------------------------
# 测试脚本（保留）
# -----------------------------

def testXYZposition(t):
    desPos = np.array([0., 0., 0.])
    desVel = np.array([0., 0., 0.])
    desAcc = np.array([0., 0., 0.])
    desThr = np.array([0., 0., 0.])
    desEul = np.array([0., 0., 0.])
    desPQR = np.array([0., 0., 0.])
    desYawRate = 30.0*pi/180
    if 1 <= t < 4:
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
    desThr = np.array([0., 0., 0.])
    desEul = np.array([0., 0., 0.])
    desPQR = np.array([0., 0., 0.])
    desYawRate = 0.
    if 1 <= t < 4:
        desVel = np.array([3, 2, 0])
    elif t >= 4:
        desVel = np.array([3, -1, 0])
    sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)
    return sDes

# -----------------------------
# 轨迹类
# -----------------------------

class Trajectory:
    def __init__(self, quad, ctrlType, trajSelect):
        """
        参数:
        - quad: 载具对象，要求至少有属性 psi (初始偏航角)、pos (当前世界系位置)
        - ctrlType: 控制类型字符串，如 "xyz_pos", "xyz_vel", "xy_vel_z_pos"
        - trajSelect: [xyzType, yawType, averVel]
            - xyzType: 轨迹类型
            - yawType: 航向控制类型
            - averVel: 使用平均速度(=1) 或使用给定的 t_wps 时间(=0)
        """
        self.ctrlType = ctrlType
        self.xyzType = trajSelect[0]
        self.yawType = trajSelect[1]
        self.averVel = trajSelect[2]

        # 航点生成
        t_wps, wps, y_wps, v_wp = makeWaypoints()
        self.t_wps = t_wps.astype(float)
        self.wps   = wps.astype(float)
        self.y_wps = y_wps.astype(float)
        self.v_wp  = float(v_wp)

        self.end_reached = 0

        # 轨迹系数
        self.T_segment = None
        self.deriv_order = None
        self.coeff_x = None
        self.coeff_y = None
        self.coeff_z = None

        if self.ctrlType == "xyz_pos":
            self._init_time_and_coeffs()

        if self.yawType == 4:
            self.y_wps = np.zeros(len(self.t_wps), dtype=float)

        self.current_heading = float(quad.psi)

        # 期望量初始化
        self.desPos = np.zeros(3)
        self.desVel = np.zeros(3)
        self.desAcc = np.zeros(3)
        self.desThr = np.zeros(3)
        self.desEul = np.zeros(3)
        self.desPQR = np.zeros(3)
        self.desYawRate = 0.0
        self.sDes = np.hstack(
            (self.desPos, self.desVel, self.desAcc, self.desThr, self.desEul, self.desPQR, self.desYawRate)
        ).astype(float)

        # 索引状态
        self.t_idx = 0
        self.arrived = False
        self.t_arrived = 0.0

    # ------------------------
    # 初始化/构建系数相关方法
    # ------------------------

    def _init_time_and_coeffs(self):
        """初始化每段时间（T_segment），并按 xyzType 构建多项式系数。"""
        self.T_segment = np.diff(self.t_wps)
        if self.averVel == 1:
            distance_vec = self.wps[1:] - self.wps[:-1]
            distances = np.linalg.norm(distance_vec, axis=1)
            self.T_segment = distances / max(self.v_wp, 1e-9)
            self.t_wps = np.zeros(len(self.T_segment) + 1)
            self.t_wps[1:] = np.cumsum(self.T_segment)

        if 3 <= self.xyzType <= 6:
            self.deriv_order = int(self.xyzType - 2)
            self._build_min_something_coeffs(minSomethingTraj)
        elif 7 <= self.xyzType <= 9:
            self.deriv_order = int(self.xyzType - 5)
            self._build_min_something_coeffs(minSomethingTraj_stop)
        elif 10 <= self.xyzType <= 11:
            self.deriv_order = int(self.xyzType - 7)
            self._build_min_something_coeffs(minSomethingTraj_faststop)

    def _build_min_something_coeffs(self, builder_func):
        x = self.wps[:, 0]
        y = self.wps[:, 1]
        z = self.wps[:, 2]
        self.coeff_x = builder_func(x, self.T_segment, self.deriv_order)
        self.coeff_y = builder_func(y, self.T_segment, self.deriv_order)
        self.coeff_z = builder_func(z, self.T_segment, self.deriv_order)

    # ------------------------
    # 共用小工具
    # ------------------------

    def _time_loop(self, t):
        """时间周期化到 [t0, t_end)。"""
        if len(self.t_wps) <= 1:
            return float(t), 0.0
        T_total = float(self.t_wps[-1] - self.t_wps[0])
        if T_total <= 0:
            return float(t), T_total
        t_shift = t - self.t_wps[0]
        t_mod_shift = np.mod(t_shift, T_total)
        t_mod = self.t_wps[0] + t_mod_shift
        eps = 1e-9
        if abs(t_mod - self.t_wps[-1]) < eps:
            t_mod = self.t_wps[-1] - eps
        return float(t_mod), T_total

    def _locate_t_index_left(self, t_mod):
        """返回 t_wps[t_idx] <= t_mod < t_wps[t_idx+1] 的左侧索引。"""
        if t_mod <= self.t_wps[0]:
            return 0
        if t_mod >= self.t_wps[-1]:
            return -1
        return int(np.where(t_mod <= self.t_wps)[0][0] - 1)

    # ------------------------
    # 位置轨迹（私有）
    # ------------------------

    def _pos_waypoint_timed(self, t_mod):
        if len(self.t_wps) != self.wps.shape[0]:
            raise Exception("时间数组和航点数组大小不一致。")
        if (np.diff(self.t_wps) <= 0).any():
            raise Exception("时间数组未正确排序。")
        t_idx = self._locate_t_index_left(t_mod)
        desPos = self.wps[t_idx, :]
        return t_idx, desPos

    def _pos_waypoint_interp(self, t_mod):
        if len(self.t_wps) != self.wps.shape[0]:
            raise Exception("时间数组和航点数组大小不一致。")
        if (np.diff(self.t_wps) <= 0).any():
            raise Exception("时间数组未正确排序。")
        if t_mod <= self.t_wps[0]:
            return 0, self.wps[0, :]
        if t_mod >= self.t_wps[-1]:
            return -1, self.wps[-1, :]
        t_idx = self._locate_t_index_left(t_mod)
        
        # 检查 T_segment 是否为 None
        if self.T_segment is None:
            raise Exception("T_segment 未初始化，无法进行插值计算。")
        
        scale = (t_mod - self.t_wps[t_idx]) / self.T_segment[t_idx]
        desPos = (1 - scale) * self.wps[t_idx, :] + scale * self.wps[t_idx + 1, :]
        return t_idx, desPos

    def _pos_waypoint_min(self, t_mod):
        if len(self.t_wps) != self.wps.shape[0]:
            raise Exception("时间数组和航点数组大小不一致。")
        
        # 检查 deriv_order 是否为 None
        if self.deriv_order is None:
            raise Exception("deriv_order 未初始化，无法进行最小导数轨迹计算。")
        
        nb_coeff = self.deriv_order * 2
        
        if t_mod <= self.t_wps[0]:
            t_idx = 0
            desPos = self.wps[0, :]
            desVel = np.zeros(3)
            desAcc = np.zeros(3)
            return t_idx, desPos, desVel, desAcc
        if t_mod >= self.t_wps[-1]:
            t_idx = -1
            desPos = self.wps[-1, :]
            desVel = np.zeros(3)
            desAcc = np.zeros(3)
            return t_idx, desPos, desVel, desAcc

        t_idx = self._locate_t_index_left(t_mod)
        tau = (t_mod - self.t_wps[t_idx])
        start = nb_coeff * t_idx
        end = nb_coeff * (t_idx + 1)

        t0 = get_poly_cc(nb_coeff, 0, tau)
        t1 = get_poly_cc(nb_coeff, 1, tau)
        t2 = get_poly_cc(nb_coeff, 2, tau)

        # 检查系数数组是否为 None
        if self.coeff_x is None or self.coeff_y is None or self.coeff_z is None:
            raise Exception("轨迹系数未初始化，无法计算期望状态。")

        desPos = np.array([
            self.coeff_x[start:end].dot(t0),
            self.coeff_y[start:end].dot(t0),
            self.coeff_z[start:end].dot(t0),
        ])
        desVel = np.array([
            self.coeff_x[start:end].dot(t1),
            self.coeff_y[start:end].dot(t1),
            self.coeff_z[start:end].dot(t1),
        ])
        desAcc = np.array([
            self.coeff_x[start:end].dot(t2),
            self.coeff_y[start:end].dot(t2),
            self.coeff_z[start:end].dot(t2),
        ])
        return t_idx, desPos, desVel, desAcc

    def _pos_waypoint_arrived(self, t, quad):
        dist_consider_arrived = 0.2
        if t == 0:
            self.t_idx = 0
            self.end_reached = 0
        else:
            distance_to_next_wp = np.linalg.norm(self.wps[self.t_idx, :] - quad.pos)
            if distance_to_next_wp < dist_consider_arrived:
                self.t_idx += 1
                if self.t_idx >= len(self.wps[:, 0]):
                    self.t_idx = 0
                    self.end_reached = 0
        desPos = self.wps[self.t_idx, :]
        return self.t_idx, desPos

    def _pos_waypoint_arrived_wait(self, t, quad):
        dist_consider_arrived = 0.2
        if t == 0:
            self.t_idx = 0
            self.t_arrived = 0.0
            self.arrived = True
            self.end_reached = 0
        else:
            distance_to_next_wp = np.linalg.norm(self.wps[self.t_idx, :] - quad.pos)
            if (distance_to_next_wp < dist_consider_arrived) and not self.arrived:
                self.t_arrived = t
                self.arrived = True
            elif self.arrived and (t - self.t_arrived > self.t_wps[self.t_idx]):
                self.t_idx += 1
                self.arrived = False
                if self.t_idx >= len(self.wps[:, 0]):
                    self.t_idx = 0
                    self.end_reached = 0
        desPos = self.wps[self.t_idx, :]
        return self.t_idx, desPos

    # ------------------------
    # 航向轨迹
    # ------------------------

    def _yaw_waypoint_timed(self, t_idx):
        if len(self.t_wps) != len(self.y_wps):
            raise Exception("时间数组和偏航数组大小不一致。")
        desYaw = self.y_wps[t_idx]
        return desYaw

    def _yaw_waypoint_interp(self, t_mod, t_idx):
        if len(self.t_wps) != len(self.y_wps):
            raise Exception("时间数组和偏航数组大小不一致。")
        if (t_mod <= self.t_wps[0]) or (t_mod >= self.t_wps[-1]):
            return self.y_wps[t_idx]
        
        # 检查 T_segment 是否为 None
        if self.T_segment is None:
            raise Exception("T_segment 未初始化，无法进行航向插值计算。")
        
        scale = (t_mod - self.t_wps[t_idx]) / self.T_segment[t_idx]
        desYaw = (1 - scale) * self.y_wps[t_idx] + scale * self.y_wps[t_idx + 1]
        return desYaw

    def _yaw_follow(self, t, t_mod, Ts, quad, xyzType, desPos, desVel, t_idx):
        if xyzType in (1, 2, 12):
            desYaw = 0.0 if t == 0 else np.arctan2(desPos[1] - quad.pos[1], desPos[0] - quad.pos[0])
        else:
            if (t_mod <= self.t_wps[0]) or (t_mod >= self.t_wps[-1]):
                desYaw = self.y_wps[t_idx]
            else:
                desYaw = np.arctan2(desVel[1], desVel[0])
        if (np.sign(desYaw) - np.sign(self.current_heading)) and abs(float(desYaw) - float(self.current_heading)) >= 2 * pi - 0.1:
            self.current_heading = self.current_heading + np.sign(desYaw) * 2 * pi
        desYawRate = (desYaw - self.current_heading) / Ts
        self.current_heading = desYaw
        return desYaw, desYawRate

    # ------------------------
    # 特殊轨迹
    # ------------------------
    # stage = 0
    def altitude_vel_random_step(self, t, z=1.0):
        """定高 + 速度随机阶跃（幅值-5~+5，阶跃时间3秒，简化版）"""
        desPos = np.array([0., 0., z])
        desAcc = np.zeros(3)
        desThr = np.zeros(3)
        desEul = np.zeros(3)
        desPQR = np.zeros(3)
        desYawRate = 0.0

        if t < 1.0:
            desVel = np.zeros(3)
        else:
            # 用段索引控制每 3 秒更新一次，但没有缓存，需固定随机种子或外部控制调用频次
            seg = int(np.floor((t - 1.0) / 3.0))
            rng = np.random.default_rng(seg)  # 用 seg 当种子，确保同一段内一致
            desVel = rng.uniform(-5.0, 5.0, size=3)
            desVel[2] = 0.0  # 若不需要 Z 速度

        sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)
        return sDes

    # stage = 1
    def altitude_vel_sweep_amp_freq(self, t, phase_offset=0.0, sweep_f=None,
                          amp_f_vx=None, amp_f_vy=None, amp_f_vz=None):
        """定高 + 速度扫频/扫幅；并给出一致的加速度参考（速度导数）。"""
        desPos = np.zeros(3)
        desVel = np.zeros(3)
        desAcc = np.zeros(3)
        desThr = np.zeros(3)
        desEul = np.zeros(3)
        desPQR = np.zeros(3)
        desYawRate = 0.0

        # 检查 sweep_f 参数是否为 None 并转换为数值
        f_val = sweep_f(t) if callable(sweep_f) else 0.25
        if f_val is None:
            f_val = 0.25
        # 确保 f_val 是数值类型
        if not isinstance(f_val, (int, float, np.number)):
            f_val = 0.25
        w = 2*np.pi*float(f_val)

        # 检查幅值函数返回值并转换为数值
        amp_vx_val = amp_f_vx(t) if callable(amp_f_vx) else 1.0
        amp_vy_val = amp_f_vy(t) if callable(amp_f_vy) else 1.0 
        amp_vz_val = amp_f_vz(t) if callable(amp_f_vz) else 0.0
        
        if amp_vx_val is None or not isinstance(amp_vx_val, (int, float, np.number)):
            amp_vx_val = 1.0
        if amp_vy_val is None or not isinstance(amp_vy_val, (int, float, np.number)):
            amp_vy_val = 1.0
        if amp_vz_val is None or not isinstance(amp_vz_val, (int, float, np.number)):
            amp_vz_val = 0.0
            
        amp_vx = float(amp_vx_val)
        amp_vy = float(amp_vy_val)
        amp_vz = float(amp_vz_val)

        vx = amp_vx * np.sin(w*t + phase_offset + 0.0)
        vy = amp_vy * np.sin(w*t + phase_offset + np.pi/2.0)
        vz = amp_vz * np.sin(w*t + phase_offset + 0.0)
        desVel[:] = [vx, vy, vz]

        ax = amp_vx * w * np.cos(w*t + phase_offset + 0.0)
        ay = amp_vy * w * np.cos(w*t + phase_offset + np.pi/2.0)
        az = amp_vz * w * np.cos(w*t + phase_offset + 0.0)
        desAcc[:] = [ax, ay, az]

        sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)
        return sDes
    
    # stage = 2
    def altitude_pos_figure8(self, t, z=1.0,
                            A=3.0, B=1.0, omega=0.4, phase=np.pi/2,
                            center=np.array([0.0, 0.0])):
        """8字航线（李萨如型），定高 z
        x = A * sin(omega * t)
        y = B * sin(2 * omega * t + phase)

        参数:
            t: 时间（s）
            z: 定高（m）
            A, B: x/y 半幅
            omega: 基础角频率（rad/s）
            phase: 相位，用于控制8字的交叉角度（默认 pi/2）
            center: 路径中心 [cx, cy]
        返回:
            sDes: np.array([...]) = [desPos(3), desVel(3), desAcc(3), desThr(3), desEul(3), desPQR(3), desYawRate(1)]
        """
        # 位置
        x = A * np.sin(omega * t) + center[0]
        y = B * np.sin(2 * omega * t + phase) + center[1]
        desPos = np.array([x, y, z], dtype=float)

        # 速度
        vx = A * omega * np.cos(omega * t)
        vy = B * 2 * omega * np.cos(2 * omega * t + phase)
        vz = 0.0
        desVel = np.array([vx, vy, vz], dtype=float)

        # 加速度
        ax = -A * (omega**2) * np.sin(omega * t)
        ay = -B * (2 * omega)**2 * np.sin(2 * omega * t + phase)
        az = 0.0
        desAcc = np.array([ax, ay, az], dtype=float)

        # 其他期望量（保持 0）
        desThr = np.zeros(3, dtype=float)
        eps = 1e-6
        psi = np.arctan2(desVel[1], desVel[0]) if np.hypot(desVel[0], desVel[1]) > eps else 0.0

        
        desEul = np.array([0.0, 0.0, psi], dtype=float)
        desPQR = np.zeros(3, dtype=float)
        desYawRate = 0.0
        sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)
        return sDes

    # stage = 3 : 复杂样条
    def pos_step_circle_figure8(self, t, z=1.0,
                                # --- 段 1：平滑“阶跃”参数（直线样条）---
                                step_start=np.array([0.0, 0.0]),  # 起点 (x0, y0)
                                step_end=np.array([5.0, 2.0]),    # 终点 (x1, y1)
                                T1=4.0,                           # 段1时长（s）
                                # --- 段 2：圆弧参数 ---
                                circle_center=np.array([5.0, 2.0]),
                                R=3.0,                            # 半径（m）
                                omega_c=0.6,                      # 圆弧角速度（rad/s）
                                arc_angle=np.pi,                  # 扫过角度：默认半圆（rad）
                                ccw=True,                         # True逆时针，False顺时针
                                # --- 段 3：8字（李萨如）参数 ---
                                A=4.0, B=2.5,                     # x/y 半幅
                                omega_f8=0.5,                     # 基础角频率（rad/s）
                                phase=np.pi/2,                    # 相位
                                fig8_center=np.array([5.0, -1.0]),
                                T3=None                           # 段3时长；None 则按整数个周期自动对齐
                                ):
        """
        复杂样条曲线：段1(平滑直线样条)->段2(圆弧)->段3(李萨如8字)，循环往复
        输出 sDes = [desPos(3), desVel(3), desAcc(3), desThr(3), desEul(3), desPQR(3), desYawRate(1)]
        """

        # ---------- 工具函数 ----------
        def quintic_blend(s):
            # 五次多项式时间缩放：0->1，且 0/1 处 v,a 为 0
            return 10*s**3 - 15*s**4 + 6*s**5

        def clamp01(x):
            return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

        def unwrap(prev, curr):
            if prev is None:
                return curr
            d = curr - prev
            d = (d + np.pi) % (2*np.pi) - np.pi
            return prev + d

        # ---------- 段时长 ----------
        # 段2时长由角速度和扫角决定
        T2 = abs(arc_angle) / max(1e-6, abs(omega_c))

        # 段3缺省时长：按 8 字的整数周期，避免段切换时的相位不连续
        if T3 is None:
            # 李萨如：x = A sin(w t), y = B sin(2 w t + phase) 的基本重复周期为 2π/w
            T3 = 2*np.pi / max(1e-6, abs(omega_f8))

        # 总周期
        T_total = T1 + T2 + T3

        # 将时间映射到一个周期内
        t_mod = np.mod(t, T_total)

        # ---------- 输出占位 ----------
        x = y = vx = vy = ax = ay = 0.0

        # ---------- 段1：平滑“阶跃”（五次多项式样条连接） ----------
        if t_mod < T1:
            s = clamp01(t_mod / T1)  # 归一化时间
            b = quintic_blend(s)
            db_ds = 30*s**2 - 60*s**3 + 30*s**4
            d2b_ds2 = 60*s - 180*s**2 + 120*s**3

            p0 = np.asarray(step_start, dtype=float)
            p1 = np.asarray(step_end, dtype=float)
            dp = p1 - p0

            # 位置
            pos_xy = p0 + b * dp
            x, y = pos_xy[0], pos_xy[1]

            # s_dot = 1/T1, s_ddot = 0
            s_dot = 1.0 / T1
            s_ddot = 0.0

            # 速度/加速度（链式法则）
            vx, vy = (db_ds * s_dot) * dp
            ax, ay = (d2b_ds2 * (s_dot**2) + db_ds * s_ddot) * dp

        # ---------- 段2：圆弧 ----------
        elif t_mod < T1 + T2:
            t2 = t_mod - T1
            # 从圆心构造初始角：指向 step_end 的角度
            cx, cy = float(circle_center[0]), float(circle_center[1])
            x0, y0 = float(step_end[0]), float(step_end[1])
            theta0 = np.arctan2(y0 - cy, x0 - cx)  # 起始角

            # 角度演化
            dir_sign = 1.0 if ccw else -1.0
            theta = theta0 + dir_sign * omega_c * t2
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            # 位置
            x = cx + R * cos_t
            y = cy + R * sin_t

            # 速度（切向）
            vx = -R * omega_c * sin_t * dir_sign
            vy =  R * omega_c * cos_t * dir_sign

            # 加速度（向心）
            ax = -R * (omega_c**2) * cos_t
            ay = -R * (omega_c**2) * sin_t

        # ---------- 段3：李萨如 8 字 ----------
        else:
            t3 = t_mod - (T1 + T2)
            cx8, cy8 = float(fig8_center[0]), float(fig8_center[1])

            # 位置
            x = cx8 + A * np.sin(omega_f8 * t3)
            y = cy8 + B * np.sin(2 * omega_f8 * t3 + phase)

            # 速度
            vx = A * omega_f8 * np.cos(omega_f8 * t3)
            vy = B * 2 * omega_f8 * np.cos(2 * omega_f8 * t3 + phase)

            # 加速度
            ax = -A * (omega_f8**2) * np.sin(omega_f8 * t3)
            ay = -B * (2 * omega_f8)**2 * np.sin(2 * omega_f8 * t3 + phase)

        # ---------- 组装期望量 ----------
        desPos = np.array([x, y, z], dtype=float)
        desVel = np.array([vx, vy, 0.0], dtype=float)
        desAcc = np.array([ax, ay, 0.0], dtype=float)

        # 航向与速度方向对齐 + 相位展开
        psi_raw = np.arctan2(vy, vx) if (vx*vx + vy*vy) > 1e-14 else 0.0

        # 在函数属性中缓存（单函数即可用），注意多实例时请改为类成员缓存
        if not hasattr(self, "_pos_spline_prev_psi"):
            self._pos_spline_prev_psi = None
        psi = unwrap(self._pos_spline_prev_psi, psi_raw)
        self._pos_spline_prev_psi = psi

        desEul = np.array([0.0, 0.0, psi], dtype=float)

        # yaw rate 解析式：ψ_dot = (vx*ay - vy*ax) / (vx^2 + vy^2)
        denom = vx*vx + vy*vy
        desYawRate = (vx * ay - vy * ax) / denom if denom > 1e-12 else 0.0

        desThr = np.zeros(3, dtype=float)
        desPQR = np.zeros(3, dtype=float)

        sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)
        return sDes
        
    # stage = 
    def pos_spline_3dspiral(self, t):
        """三维螺旋线样条轨迹，航向跟随速度方向并做相位展开"""
        # 预设
        desThr = np.zeros(3, dtype=float)
        desPQR = np.zeros(3, dtype=float)

        # 轨迹参数
        A = 5.0   # 半径
        b = 1.0   # 螺距系数（z = b * w * t）
        f = 0.05  # 圆频率（Hz）
        w = 2 * np.pi * f

        # 位置
        x = A * np.cos(w * t)
        y = A * np.sin(w * t)
        z = b * w * t
        desPos = np.array([x, y, z], dtype=float)

        # 速度
        vx = -A * w * np.sin(w * t)
        vy =  A * w * np.cos(w * t)
        vz =  b * w
        desVel = np.array([vx, vy, vz], dtype=float)

        # 加速度
        ax = -A * w**2 * np.cos(w * t)
        ay = -A * w**2 * np.sin(w * t)
        az =  0.0
        desAcc = np.array([ax, ay, az], dtype=float)

        # 航向：与水平速度方向对齐（忽略 vz）
        speed_xy2 = vx*vx + vy*vy
        psi_raw = np.arctan2(vy, vx) if speed_xy2 > 1e-12 else (getattr(self, "_prev_psi_spiral", 0.0))

        # 相位展开（保持航向连续）
        prev = getattr(self, "_prev_psi_spiral", None)
        if prev is None:
            psi = psi_raw
        else:
            d = psi_raw - prev
            d = (d + np.pi) % (2*np.pi) - np.pi
            psi = prev + d
        self._prev_psi_spiral = psi

        desEul = np.array([0.0, 0.0, psi], dtype=float)

        # 解析偏航角速度 ψ_dot
        desYawRate = (vx * ay - vy * ax) / speed_xy2 if speed_xy2 > 1e-12 else 0.0

        sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)
        return sDes
    
    # stage = 4
    def pos_traj_random_path(self, t):
        """
        随机路径轨迹（强波动版）：
        - 几何波动大：空间范围约 ±20m
        - 速度波动大：目标区间 [1, 15] m/s
        - 姿态约束友好：通过水平加速度限幅（~9.8 m/s^2）间接匹配 ±45°
        - Catmull-Rom 样条 + 航向随速度 + ψ 相位展开 + 解析 yaw rate
        """
        # ---------------- 配置区（可按需调整） ----------------
        # 轨迹空间尺度（位置波动）
        base_radius = 12.0            # 基础半径，结合抖动后总体幅度接近 ±20m
        radius_jitter = 10.0          # 半径扰动幅度
        height_mean = 2.0
        height_amp = 10.0             # 高度波动幅度（z 方向也有起伏）
        z_noise = 0.8                 # 高度噪声

        # 航点数量（越多越曲折）
        num_points = 18

        # 总时长（循环）
        total_time = 60.0

        # 速度波动与限幅
        v_min = 1.0
        v_max = 15.0

        # 速度起伏轮廓（非线性进度）
        speed_amp = 0.75              # 速度振幅系数（0..1，越大越剧烈）
        speed_freq = 0.10             # 速度起伏频率（Hz）
        speed_noise_scale = 0.20      # 速度噪声强度
        profile_table_N = 2048        # 速度进度查表的离散点数量

        # 姿态友好限制：水平加速度限幅（g * tan(45°) ≈ 9.8 m/s^2）
        a_xy_max = 9.8

        # 随机性（可固定也可不固定）
        rng = np.random.default_rng(42)

        # ---------------- 航点生成（强波动） ----------------
        # 非均匀角分布
        angles = np.sort(rng.uniform(0.0, 2*np.pi, size=num_points))
        angle_jitter = 0.6
        angles = (angles + rng.uniform(-angle_jitter, angle_jitter, size=num_points)) % (2*np.pi)

        # 半径随机（避免过小）
        r = base_radius + rng.uniform(-radius_jitter, radius_jitter, size=num_points)
        r = np.clip(r, 2.0, None)  # 防止接近 0 的极端半径

        # 高度随机（带粗糙度）
        z = height_mean \
            + height_amp * np.sin(angles * rng.uniform(1.0, 2.2)) \
            + rng.normal(0.0, z_noise, size=num_points)

        # 航点
        points = np.vstack([r * np.cos(angles),
                            r * np.sin(angles),
                            z]).T

        # 位置范围备注：以 12±10 米的半径为主，典型 xy 幅度 2~22 m，结合 Catmull-Rom 曲率，整体在 ±20 m 的量级

        # ---------------- Catmull-Rom 样条工具 ----------------
        def catmull_rom(p0, p1, p2, p3, u):
            u2, u3 = u*u, u*u*u
            # 位置
            a = -0.5*u3 + u2 - 0.5*u
            b =  1.5*u3 - 2.5*u2 + 1.0
            c = -1.5*u3 + 2.0*u2 + 0.5*u
            d =  0.5*u3 - 0.5*u2
            pos = a*p0 + b*p1 + c*p2 + d*p3
            # 一阶导（对 u）
            da = -1.5*u*u + 2.0*u - 0.5
            db =  4.5*u*u - 5.0*u
            dc = -4.5*u*u + 4.0*u + 0.5
            dd =  1.5*u*u - 1.0*u
            dpos_du = da*p0 + db*p1 + dc*p2 + dd*p3
            # 二阶导（对 u）
            dda = -3.0*u + 2.0
            ddb =  9.0*u - 5.0
            ddc = -9.0*u + 4.0
            ddd =  3.0*u - 1.0
            d2pos_du2 = dda*p0 + ddb*p1 + ddc*p2 + ddd*p3
            return pos, dpos_du, d2pos_du2

        # 扩展首尾以闭环
        pts = np.vstack([points[-1], points, points[0], points[1]])  # P_{-1}, P0..P_{n-1}, P_n, P_{n+1}

        # ---------------- 非线性时间进度（制造速度波动） ----------------
        def raw_speed_profile(t_local):
            # 产生正值的速度轮廓（未缩放），包含：基线 + 正弦 + 噪声
            base = 1.0 + speed_amp * np.sin(2*np.pi*speed_freq * t_local)
            noise = rng.normal(0.0, speed_noise_scale)
            s = max(0.1, base + noise)
            return s

        # 构建查表：t -> 归一化进度（0..1），表示走过的路径比例
        ts = np.linspace(0.0, total_time, profile_table_N)
        speeds = np.array([raw_speed_profile(tt) for tt in ts])
        cum = np.cumsum(speeds)
        cum = (cum - cum[0]) / (cum[-1] - cum[0] + 1e-12)

        def time_to_progress(t_local):
            t_local = t_local % total_time
            x = (t_local / total_time) * (profile_table_N - 1)
            i = int(np.floor(x))
            a = x - i
            i1 = min(i + 1, profile_table_N - 1)
            # 线性插值
            return (1 - a) * cum[i] + a * cum[i1]

        # 数值微分（用于 u_dot 和 u_ddot）
        def progress_derivatives(t_local, eps=1e-3):
            p = time_to_progress(t_local)
            p_plus = time_to_progress(t_local + eps)
            p_minus = time_to_progress(t_local - eps)
            dp_dt = (p_plus - p_minus) / (2*eps)
            d2p_dt2 = (time_to_progress(t_local + eps) - 2*p + time_to_progress(t_local - eps)) / (eps**2)
            return p, dp_dt, d2p_dt2

        # ---------------- 将 t 映射到段与参数 u ----------------
        # progress ∈ [0,1) 映射到 num_points 段
        progress, dprogress_dt, d2progress_dt2 = progress_derivatives(t if np.isfinite(t) else 0.0)
        prog_scaled = progress * num_points
        seg_idx = int(np.floor(prog_scaled)) % num_points
        u = prog_scaled - np.floor(prog_scaled)  # [0,1)

        # 目标速度范围缩放（把单位“进度速度”映射为实际速度）
        # 进度的线速度 v ≈ |dpos/du| * du/dt；先估计局部 |dpos/du|
        i = seg_idx + 1
        p0, p1, p2, p3 = pts[i-1], pts[i], pts[i+1], pts[i+2]
        _, dpos_du_est, d2pos_du2_est = catmull_rom(p0, p1, p2, p3, u)
        # 只考虑水平速度目标范围
        dposdu_xy = dpos_du_est[:2]
        norm_dposdu_xy = max(1e-6, np.linalg.norm(dposdu_xy))
        # 理想 du/dt 以期望速度落在 [v_min, v_max] 中部
        v_target = 0.5 * (v_min + v_max)  # 可改成随时间波动：v_target = v_min + (v_max - v_min) * progress
        du_dt_nominal = v_target / norm_dposdu_xy

        # 综合进度速度：使用 dprogress/dt 来制造波动，再缩放到目标范围
        du_dt = num_points * max(1e-6, dprogress_dt)  # 来自非线性进度
        # 归一化后映射到 [v_min, v_max]
        # 当前水平速度估计
        v_xy_est = norm_dposdu_xy * du_dt
        if v_xy_est < v_min:
            scale = (v_min / max(1e-6, v_xy_est))
            du_dt *= scale
        elif v_xy_est > v_max:
            scale = (v_max / v_xy_est)
            du_dt *= scale

        # 加速度项需要 ü
        d2u_dt2 = num_points * d2progress_dt2

        # ---------------- 样条求解位置/速度/加速度 ----------------
        pos, dpos_du, d2pos_du2 = catmull_rom(p0, p1, p2, p3, u)
        vel = dpos_du * du_dt
        acc = d2pos_du2 * (du_dt**2) + dpos_du * d2u_dt2

        # ---------------- 水平加速度限幅（姿态友好） ----------------
        # 将水平加速度限制在 a_xy_max 内，避免需要过大俯仰/横滚
        a_xy = acc[:2]
        a_xy_norm = np.linalg.norm(a_xy)
        if a_xy_norm > a_xy_max:
            acc[:2] = a_xy * (a_xy_max / a_xy_norm)

        # ---------------- 输出组装 + 航向/偏航 ----------------
        desPos = pos.astype(float)
        desVel = vel.astype(float)
        desAcc = acc.astype(float)

        # 航向：与水平速度方向一致（忽略 vz）
        vx, vy, vz = desVel
        ax, ay, az = desAcc
        speed_xy2 = vx*vx + vy*vy

        psi_raw = np.arctan2(vy, vx) if speed_xy2 > 1e-10 else (getattr(self, "_prev_psi_rand", 0.0))

        # 相位展开（避免 ±π 跳变）
        prev = getattr(self, "_prev_psi_rand", None)
        if prev is None:
            psi = psi_raw
        else:
            d_ang = psi_raw - prev
            d_ang = (d_ang + np.pi) % (2*np.pi) - np.pi
            psi = prev + d_ang
        self._prev_psi_rand = psi

        desEul = np.array([0.0, 0.0, psi], dtype=float)

        # 偏航角速度（解析）
        desYawRate = (vx * ay - vy * ax) / speed_xy2 if speed_xy2 > 1e-10 else 0.0

        # 其余量
        desThr = np.zeros(3, dtype=float)
        desPQR = np.zeros(3, dtype=float)

        sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)
        return sDes
    # ------------------------
    # 主接口：计算期望状态
    # ------------------------

    def desiredState(self, t, Ts, quad):
        """
        根据当前时间 t 和轨迹类型，计算并返回期望状态 sDes。
        返回 sDes = [desPos(3), desVel(3), desAcc(3), desThr(3), desEul(3), desPQR(3), desYawRate]
        """
        # 清空本次期望
        self.desPos[:] = 0.0
        self.desVel[:] = 0.0
        self.desAcc[:] = 0.0
        self.desThr[:] = 0.0
        self.desEul[:] = 0.0
        self.desPQR[:] = 0.0
        self.desYawRate = 0.0

        use_time_driven = (self.ctrlType == "xyz_pos") and (
            self.xyzType in [1, 2] or (3 <= self.xyzType <= 11)
        )
        if use_time_driven:
            t_mod, _ = self._time_loop(t)
        else:
            t_mod = float(t)

        # 速度控制类直接走测试函数（保留）
        if self.ctrlType == "xyz_vel":
            if self.xyzType == 1:
                self.sDes = testVelControl(t)

        elif self.ctrlType == "xy_vel_z_pos":
            # 统一相位（可选）
            phase = getattr(config, "ALT_SWEEP_PHASE", 0.0)

            # -------- Vel 模式（xyzType == 2）往返扫 --------
            vel_sweep_f = lambda tt: sine_sweep_pingpong(
                tt,
                low=getattr(config, "VEL_SWEEP_LOW_HZ", 0.0),
                high=getattr(config, "VEL_SWEEP_HIGH_HZ", 3.0),
                T=getattr(config, "VEL_SWEEP_PERIOD_S", 60.0),
                mode=getattr(config, "VEL_SWEEP_MODE", "lin"),
            )
            vel_amp_f_vx = lambda tt: amp_sweep_pingpong(
                tt,
                low=getattr(config, "VEL_AMP_VX_LOW_MS", 0.0),
                high=getattr(config, "VEL_AMP_VX_HIGH_MS", 15.0),
                T=getattr(config, "VEL_AMP_PERIOD_S", 60.0),
                mode="lin",
            )
            vel_amp_f_vy = lambda tt: amp_sweep_pingpong(
                tt,
                low=getattr(config, "VEL_AMP_VY_LOW_MS", 0.0),
                high=getattr(config, "VEL_AMP_VY_HIGH_MS", 15.0),
                T=getattr(config, "VEL_AMP_PERIOD_S", 60.0),
                mode="lin",
            )
            vel_amp_f_vz = lambda tt: amp_sweep_pingpong(
                tt,
                low=getattr(config, "VEL_AMP_VZ_LOW_MS", 0.0),
                high=getattr(config, "VEL_AMP_VZ_HIGH_MS", 0.0),
                T=getattr(config, "VEL_AMP_PERIOD_S", 60.0),
                mode="lin",
            )

            if self.xyzType == 201:
                # 课程stage = 0：定高 + 随机阶跃
                # print("ctrlType == xy_vel_z_pos, xyzType=201 未定义轨迹。")
                self.sDes = self.altitude_vel_random_step(t, z=1.0)
                pass
            elif self.xyzType == 202:
                # 课程 stage = 1：定高 + 扫频扫幅 
                # print("ctrlType == xy_vel_z_pos, xyzType=202 未定义轨迹。")
                self.sDes = self.altitude_vel_sweep_amp_freq(
                    t,
                    phase_offset=phase,
                    sweep_f=vel_sweep_f,
                    amp_f_vx=vel_amp_f_vx,
                    amp_f_vy=vel_amp_f_vy,
                    amp_f_vz=vel_amp_f_vz,
                )

        elif self.ctrlType == "xyz_pos":
            if self.xyzType == 0: 
                # 定点飞行
                print("ctrlType == xyz_pos, xyzType=0 未定义轨迹。")
                pass
            elif self.xyzType == 99:
                self.sDes = testXYZposition(t)
            elif self.xyzType == 101:            
                # 课程stage = 2：8字航线
                # print("ctrlType == xyz_pos, xyzType=101 未定义轨迹。")
                self.sDes = self.altitude_pos_figure8(t)
                pass
            elif self.xyzType == 102:            
                # 课程stage = 3：复杂样条
                # print("ctrlType == xyz_pos, xyzType=102 未定义轨迹。")
                self.sDes = self.pos_step_circle_figure8(t)
                pass
            elif self.xyzType == 103:            
                # 课程stage = 4：随机轨迹
                # print("ctrlType == xyz_pos, xyzType=103 未定义轨迹。")
                self.sDes = self.pos_traj_random_path(t)
                pass
            else:                               # 可以选一个轨迹（xyzType=5）用于在线强化学习测试 （课程stage = 5）
                # 位置轨迹类型
                if self.xyzType == 1:
                    self.t_idx, self.desPos = self._pos_waypoint_timed(t_mod)
                elif self.xyzType == 2:
                    self.t_idx, self.desPos = self._pos_waypoint_interp(t_mod)
                elif 3 <= self.xyzType <= 11:
                    self.t_idx, self.desPos, self.desVel, self.desAcc = self._pos_waypoint_min(t_mod)
                elif self.xyzType == 12:
                    self.t_idx, self.desPos = self._pos_waypoint_arrived(t, quad)
                elif self.xyzType == 13:
                    self.t_idx, self.desPos = self._pos_waypoint_arrived_wait(t, quad)

                # 航向轨迹类型
                if self.yawType == 0:
                    pass
                elif self.yawType == 1:
                    self.desEul[2] = self._yaw_waypoint_timed(self.t_idx)
                elif self.yawType == 2:
                    self.desEul[2] = self._yaw_waypoint_interp(t_mod, self.t_idx)
                elif self.yawType == 3:
                    self.desEul[2], self.desYawRate = self._yaw_follow(
                        t=t,
                        t_mod=t_mod,
                        Ts=Ts,
                        quad=quad,
                        xyzType=self.xyzType,
                        desPos=self.desPos,
                        desVel=self.desVel,
                        t_idx=self.t_idx,
                    )

                self.sDes = np.hstack(
                    (self.desPos, self.desVel, self.desAcc, self.desThr, self.desEul, self.desPQR, self.desYawRate)
                ).astype(float)

        return self.sDes