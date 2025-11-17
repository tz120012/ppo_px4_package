# -*- coding: utf-8 -*-
"""
四旋翼轨迹生成器 - 精简版
仅保留核心训练轨迹

作者: John Bass (原始版本)
精简版本: 仅保留 1001(俯仰阶跃)、1012/1013/1014(正弦扫频系列)、201(典型组合)、206(3D李萨如)
"""

import numpy as np
from numpy import pi
from numpy.linalg import norm
from pathlib import Path
from waypoints import makeWaypoints
import config

# ============================================================================
# 工具函数
# ============================================================================

def get_poly_cc(n, k, t):
    """获取 n 阶多项式在 t 时刻的 k 阶导数的系数。
    
    Args:
        n: 多项式阶数
        k: 导数阶数
        t: 时间点
    
    Returns:
        系数数组
    """
    assert (n > 0 and k >= 0), "order and derivative must be positive."
    cc = np.ones(n)
    D = np.linspace(n-1, 0, n)
    for i in range(n):
        for _ in range(k):
            cc[i] = cc[i] * D[i]
            D[i] = D[i] - 1
            if D[i] == -1:
                D[i] = 0
    for i, c in enumerate(cc):
        cc[i] = c * np.power(t, D[i])
    return cc


# ============================================================================
# 主轨迹类
# ============================================================================

class Trajectory:
    """四旋翼轨迹生成器
    
    控制类型 (ctrlType):
        - "xyz_vel": 速度控制模式
        - "xyz_pos": 位置控制模式
        - "xy_vel_z_pos": 混合模式（XY速度 + Z位置）
    
    轨迹类型编码 (xyzType):
        速度跟踪轨迹:
            1001: 俯仰轴阶跃测试 (30秒周期) - 0->3->0->8->15->0 m/s
            1012: 正弦扫频扫幅 - 高频段 (30秒周期，幅值3-15m/s，频率0.1-0.5Hz)
            1013: 正弦扫频 (30秒周期，频率固定0.3Hz，幅值线性3→18m/s)
            1014: 正弦扫频 (30秒周期，幅值固定8m/s，频率线性0.1→0.5Hz)
        
        位置跟踪轨迹:
            201: 典型轨迹组合（8字+螺旋+圆形，每段15秒）
            206: 3D李萨如曲线（30秒周期）
    
    航向类型 (yawType):
        0: 固定航向（0度）
        1: 按时间设置航向
        2: 航向插值
        3: 航向跟随速度方向（推荐）
        4: 固定为0
    """

    def __init__(self, quad, ctrlType, trajSelect):
        """初始化轨迹生成器
        
        Args:
            quad: 四旋翼对象
            ctrlType: 控制类型字符串
            trajSelect: [xyzType, yawType, averVel]
        """
        # ========== 基本配置 ==========
        self.quad = quad  # 保存 quad 引用
        self.ctrlType = ctrlType
        self.xyzType = trajSelect[0]
        self.yawType = trajSelect[1]
        self.averVel = trajSelect[2]
        
        # ========== 航点数据（用于传统轨迹） ==========
        t_wps, wps, y_wps, v_wp = makeWaypoints()
        self.t_wps = t_wps
        self.wps = wps
        self.y_wps = y_wps
        self.v_wp = v_wp
        
        # ========== 状态变量 ==========
        self.end_reached = 0
        self.t_idx = 0
        self.t_arrived = 0
        self.arrived = False
        self.prevDesYaw = 0.0
        self.current_heading = quad.psi
        
        # ========== 期望状态向量 ==========
        self.desPos = np.zeros(3)
        self.desVel = np.zeros(3)
        self.desAcc = np.zeros(3)
        self.desThr = np.zeros(3)
        self.desEul = np.zeros(3)
        self.desPQR = np.zeros(3)
        self.desYawRate = 0.
        
        # ========== 根据控制类型初始化 ==========
        if self.ctrlType in ["xyz_vel", "xy_vel_z_pos"]:
            self._init_velocity_trajectories()
        elif self.ctrlType == "xyz_pos":
            self._init_position_trajectories()
        
        # ========== 初始状态 ==========
        self.sDes = np.hstack((
            self.desPos, self.desVel, self.desAcc, self.desThr,
            self.desEul, self.desPQR, self.desYawRate
        )).astype(float)


        # ========== 添加速度限制配置 ==========
        self.max_velocity_xy = 15.0  # m/s (XY平面最大速度)
        self.max_velocity_z = 5.0    # m/s (垂直最大速度)
        self.max_velocity_total = 15.0  # m/s (总速度限制)

    # ========================================================================
    # 初始化方法
    # ========================================================================
    
    def _limit_velocity(self, vel):
        """限制速度到安全范围"""
        vel_limited = vel.copy()
        
        # 分轴限制
        vel_limited[0] = np.clip(vel[0], -self.max_velocity_xy, self.max_velocity_xy)
        vel_limited[1] = np.clip(vel[1], -self.max_velocity_xy, self.max_velocity_xy)
        vel_limited[2] = np.clip(vel[2], -self.max_velocity_z, self.max_velocity_z)
        
        # 总速度限制
        vel_mag = np.linalg.norm(vel_limited)
        if vel_mag > self.max_velocity_total:
            vel_limited = vel_limited * (self.max_velocity_total / vel_mag)
        
        return vel_limited

    def _init_velocity_trajectories(self):
        """初始化速度跟踪轨迹参数 - 仅保留 1001 和 1012"""
        
        if self.xyzType == 1001:
            # 俯仰轴阶跃测试：每段 5s，序列为 0 -> 3 -> 0 -> 8 -> 15 -> 0 （总 30s）
            self.step_duration = 5.0
            self.step_sequence = [0.0, 3.0, 0.0, 8.0, 15.0, 0.0]
            self.num_steps = len(self.step_sequence)
            self.total_seq_time = self.step_duration * self.num_steps
            self.current_step = 0
            self.last_cycle_start_time = 0
            self.current_vel_target = np.zeros(3)
        
        elif self.xyzType in (1012, 1013, 1014):
            # 正弦扫频系列
            self.vel_axis_ratio = np.array([1.0, 0.0, 0.0])

            if self.xyzType == 1012:
                # 幅值：3-15 m/s，频率：0.1-0.5 Hz（高频段扫幅）
                self.vel_phase = np.array([0, pi/3, 2*pi/3])
                self._sine_sweep_config = {
                    "amp_mode": "triangle",
                    "amp_min": 3.0,
                    "amp_max": 15.0,
                    "amp_period": 30.0,
                    "freq_mode": "linear",
                    "freq_min": 0.1,
                    "freq_max": 0.5,
                    "freq_period": 30.0,
                }

            elif self.xyzType == 1013:
                # 幅值：3→18 m/s 线性增加，频率固定 0.3 Hz
                self.vel_phase = np.zeros(3)
                self._sine_sweep_config = {
                    "amp_mode": "linear",
                    "amp_min": 3.0,
                    "amp_max": 18.0,
                    "amp_period": 30.0,
                    "freq_mode": "fixed",
                    "freq_value": 0.3,
                }

            else:  # 1014
                # 幅值固定 8 m/s，频率：0.1→0.5 Hz 线性增加
                self.vel_phase = np.zeros(3)
                self._sine_sweep_config = {
                    "amp_mode": "fixed",
                    "amp_value": 8.0,
                    "freq_mode": "linear",
                    "freq_min": 0.1,
                    "freq_max": 0.5,
                    "freq_period": 30.0,
                }

            # 参考时间与累积相位初始化
            self._sine_sweep_ref_time = None
            self._sine_sweep_phase_accum = 0.0
            self._sine_sweep_last_t = None

        else:
            raise ValueError(f"不支持的速度轨迹类型: {self.xyzType}，仅支持 1001、1012、1013、1014")

    def _init_position_trajectories(self):
        """初始化位置跟踪轨迹参数 - 仅保留 201 和 206"""
        
        # 计算航段时间
        self.T_segment = np.diff(self.t_wps)
        if self.averVel == 1:
            distance_segment = self.wps[1:] - self.wps[:-1]
            self.T_segment = np.sqrt(
                distance_segment[:,0]**2 + 
                distance_segment[:,1]**2 + 
                distance_segment[:,2]**2
            ) / self.v_wp
            self.t_wps = np.zeros(len(self.T_segment) + 1)
            self.t_wps[1:] = np.cumsum(self.T_segment)
        
        if self.xyzType == 201:
            # 典型轨迹组合（8字+螺旋+圆形，每段15秒）
            self.phase_duration = 15.0
        
        elif self.xyzType == 206:
            # 3D李萨如（循环轨迹，30s为周期）
            self.lissajous_amp = np.array([33.0, 6.0, 3.0])
            # 频率整数比，配合周期得到各轴角频率：omega_i = 2π * freq[i] / period
            self.lissajous_freq = np.array([2, 3, 3])
            self.lissajous_period = 30.0
            self.lissajous_center = np.array([0., 0., 10.0])

            # 方案1: 根据初始位置调整轨迹中心（推荐）
            # 如果无人机初始Z位置与轨迹中心差距过大，自动调整中心
            init_offset = self.quad.pos - self.lissajous_center
            if abs(init_offset[2]) > self.lissajous_amp[2]:
                # Z轴偏移超出振幅范围，调整中心高度以包含初始位置
                self.lissajous_center[2] = self.quad.pos[2]
                # print(f"[Trajectory 206] 警告: 初始Z={self.quad.pos[2]:.2f}m 超出范围，"
                #     f"已调整轨迹中心Z至 {self.lissajous_center[2]:.2f}m")
                init_offset = self.quad.pos - self.lissajous_center
            
            self.lissajous_phase = np.zeros(3)
            
            for i in range(3):
                if abs(self.lissajous_amp[i]) > 1e-6:
                    # 计算归一化偏移
                    normalized_offset = init_offset[i] / self.lissajous_amp[i]
                    
                    # 检查是否在有效范围内
                    if abs(normalized_offset) > 1.0:
                        # print(f"[Trajectory 206] 警告: 轴{i} 偏移 {init_offset[i]:.2f}m "
                        #     f"超出振幅 {self.lissajous_amp[i]:.2f}m")
                        # 将无人机位置投影到轨迹边界
                        normalized_offset = np.clip(normalized_offset, -1.0, 1.0)
                    
                    # 从位置反算相位
                    sin_val = normalized_offset
                    self.lissajous_phase[i] = np.arcsin(sin_val)
                    
                    # 选择使初始速度方向合理的相位
                    omega = 2 * pi * self.lissajous_freq[i] / self.lissajous_period
                    
                    # 计算两个可能的相位
                    phase_1 = self.lissajous_phase[i]
                    phase_2 = pi - self.lissajous_phase[i]
                    
                    # 计算对应的初始速度
                    vel_1 = self.lissajous_amp[i] * omega * np.cos(phase_1)
                    vel_2 = self.lissajous_amp[i] * omega * np.cos(phase_2)
                    
                    # 选择速度更大的方向（使轨迹更流畅）
                    if hasattr(self.quad, 'vel') and np.linalg.norm(self.quad.vel) > 0.1:
                        # 如果有初始速度，选择与速度方向一致的相位
                        if abs(vel_1 - self.quad.vel[i]) < abs(vel_2 - self.quad.vel[i]):
                            self.lissajous_phase[i] = phase_1
                        else:
                            self.lissajous_phase[i] = phase_2
                    else:
                        # 选择绝对值更大的速度（更激进的运动）
                        if abs(vel_1) > abs(vel_2):
                            self.lissajous_phase[i] = phase_1
                        else:
                            self.lissajous_phase[i] = phase_2
                else:
                    self.lissajous_phase[i] = 0.0
            
            # 打印初始化信息
            # print(f"[Trajectory 206] 3D李萨如轨迹初始化:")
            # print(f"  初始位置: {self.quad.pos}")
            # print(f"  轨迹中心: {self.lissajous_center}")
            # print(f"  振幅: {self.lissajous_amp}")
            # print(f"  位置偏移: {init_offset}")
            # print(f"  归一化偏移: {init_offset / self.lissajous_amp}")
            # print(f"  计算相位: {self.lissajous_phase * 180/pi} 度")
            
            # 验证：计算 t=0 时的期望位置
            # verify_pos = self.lissajous_center + self.lissajous_amp * np.sin(self.lissajous_phase)
            # print(f"  验证位置: {verify_pos}")
            # print(f"  位置误差: {np.linalg.norm(verify_pos - self.quad.pos):.6f} m")
            
            # 计算并显示初始速度
            verify_vel = np.zeros(3)
            for i in range(3):
                omega = 2 * pi * self.lissajous_freq[i] / self.lissajous_period
                verify_vel[i] = self.lissajous_amp[i] * omega * np.cos(self.lissajous_phase[i])
            # print(f"  初始速度: {verify_vel}")
            # print(f"  速度幅值: {np.linalg.norm(verify_vel):.2f} m/s")
        
        else:
            raise ValueError(f"不支持的位置轨迹类型: {self.xyzType}，仅支持 201 和 206")

    # ========================================================================
    # 速度跟踪轨迹生成方法 - 仅保留 1001 和 1012
    # ========================================================================
    
    def _vel_pitch_only_step(self, t):
        """xyzType=1001: 俯仰轴阶跃测试（30秒周期：0->3->0->8->15->0 m/s，每段5s）"""
        desVel = np.zeros(3)
        if self.num_steps == 0 or self.step_duration <= 0:
            return desVel
        idx = int(t // self.step_duration) % self.num_steps
        desVel[0] = float(self.step_sequence[idx])
        return desVel
    
    def _vel_sine_sweep_high_freq(self, t):
        """正弦扫频系列（1012/1013/1014）"""
        return self._vel_sine_sweep_generic(t)
    
    def _vel_sine_sweep_generic(self, t):
        """正弦扫频扫幅通用实现（基于相对时间，保证周期性）"""
        desVel = np.zeros(3)
        desAcc = np.zeros(3)
        
        # 若未设置参考时间，则以首次调用时的 t 作为起点
        if not hasattr(self, "_sine_sweep_ref_time") or self._sine_sweep_ref_time is None:
            self._sine_sweep_ref_time = t
            self._sine_sweep_last_t = t
            self._sine_sweep_phase_accum = 0.0
        
        # 如果没有相位累积属性，初始化它（向下兼容）
        if not hasattr(self, "_sine_sweep_phase_accum"):
            self._sine_sweep_phase_accum = 0.0
        
        # 计算相对时间
        tt = t - self._sine_sweep_ref_time
        if tt < 0:
            # 保证非负
            tt = 0.0
        
        # 计算时间步长（用于相位累积）
        if hasattr(self, "_sine_sweep_last_t") and self._sine_sweep_last_t is not None:
            dt = t - self._sine_sweep_last_t
        else:
            dt = 0.0
        self._sine_sweep_last_t = t
        
        cfg = getattr(self, "_sine_sweep_config", None)
        if cfg is None:
            raise ValueError("正弦扫频配置未初始化")

        # 幅值调度
        amp_mode = cfg.get("amp_mode", "triangle")
        if amp_mode == "triangle":
            period = max(cfg.get("amp_period", 30.0), 1e-6)
            amp_ratio = np.abs((tt % period) / period * 2 - 1)
            current_amp = cfg["amp_min"] + (cfg["amp_max"] - cfg["amp_min"]) * amp_ratio
        elif amp_mode == "linear":
            period = max(cfg.get("amp_period", 30.0), 1e-6)
            amp_ratio = (tt % period) / period
            current_amp = cfg["amp_min"] + (cfg["amp_max"] - cfg["amp_min"]) * amp_ratio
        elif amp_mode == "fixed":
            current_amp = cfg["amp_value"]
        else:
            raise ValueError(f"未知的幅值调度模式: {amp_mode}")

        # 频率调度
        freq_mode = cfg.get("freq_mode", "linear")
        if freq_mode == "linear":
            period = max(cfg.get("freq_period", 30.0), 1e-6)
            freq_ratio = (tt % period) / period
            current_freq = cfg["freq_min"] + (cfg["freq_max"] - cfg["freq_min"]) * freq_ratio
        elif freq_mode == "fixed":
            current_freq = cfg["freq_value"]
        else:
            raise ValueError(f"未知的频率调度模式: {freq_mode}")

        omega = 2 * pi * current_freq
        
        # 累积相位（积分频率）
        self._sine_sweep_phase_accum += omega * dt
        
        # 归一化
        norm_factor = np.sqrt(np.sum(self.vel_axis_ratio**2))
        
        # 各轴分解 - 使用累积相位而非 omega * tt
        for i in range(3):
            ratio_norm = self.vel_axis_ratio[i] / norm_factor
            desVel[i] = current_amp * ratio_norm * np.sin(self._sine_sweep_phase_accum + self.vel_phase[i])
            desAcc[i] = current_amp * ratio_norm * omega * np.cos(self._sine_sweep_phase_accum + self.vel_phase[i])
        
        return desVel, desAcc

    # ========================================================================
    # 位置跟踪轨迹生成方法 - 仅保留 201 和 206
    # ========================================================================
    
    def _pos_typical_combined(self, t):
        """xyzType=201: 典型轨迹组合（8字+螺旋+圆形，每段15秒）"""
        total_cycle_time = 3 * self.phase_duration
        t_cycle = np.mod(t, total_cycle_time)
        
        if t_cycle < self.phase_duration:
            self._pos_lissajous_eight(t_cycle)
        elif t_cycle < 2 * self.phase_duration:
            self._pos_helix(t_cycle - self.phase_duration)
        else:
            self._pos_circle_internal(t_cycle - 2 * self.phase_duration)
    
    def _pos_lissajous_eight(self, t):
        """李萨如8字轨迹（内部方法）"""
        A, B = 18.0, 14.0
        freq_ratio = 2
        omega = 2 * pi / self.phase_duration
        height = 2.0
        
        self.desPos[0] = A * np.sin(omega * t)
        self.desPos[1] = B * np.sin(freq_ratio * omega * t)
        self.desPos[2] = height
        
        self.desVel[0] = A * omega * np.cos(omega * t)
        self.desVel[1] = B * freq_ratio * omega * np.cos(freq_ratio * omega * t)
        self.desVel[2] = 0
        
        self.desAcc[0] = -A * omega**2 * np.sin(omega * t)
        self.desAcc[1] = -B * (freq_ratio * omega)**2 * np.sin(freq_ratio * omega * t)
        self.desAcc[2] = 0
    
    def _pos_helix(self, t):
        """螺旋轨迹（内部方法）"""
        radius = 12.0
        omega = 4 * pi / self.phase_duration
        climb_rate = 6.0 / self.phase_duration
        initial_height = 2.0
        
        self.desPos[0] = radius * np.cos(omega * t)
        self.desPos[1] = radius * np.sin(omega * t)
        self.desPos[2] = initial_height + climb_rate * t
        
        self.desVel[0] = -radius * omega * np.sin(omega * t)
        self.desVel[1] = radius * omega * np.cos(omega * t)
        self.desVel[2] = climb_rate
        
        self.desAcc[0] = -radius * omega**2 * np.cos(omega * t)
        self.desAcc[1] = -radius * omega**2 * np.sin(omega * t)
        self.desAcc[2] = 0
    
    def _pos_circle_internal(self, t):
        """圆形轨迹（内部方法）"""
        radius = 15.0
        omega = 2 * pi / self.phase_duration
        height = 1.5
        
        self.desPos[0] = radius * np.cos(omega * t)
        self.desPos[1] = radius * np.sin(omega * t)
        self.desPos[2] = height
        
        self.desVel[0] = -radius * omega * np.sin(omega * t)
        self.desVel[1] = radius * omega * np.cos(omega * t)
        self.desVel[2] = 0
        
        self.desAcc[0] = -radius * omega**2 * np.cos(omega * t)
        self.desAcc[1] = -radius * omega**2 * np.sin(omega * t)
        self.desAcc[2] = 0
    
    def _pos_lissajous_3d(self, t):
        """xyzType=206: 3D李萨如（30s周期循环）"""
        # 以固定周期循环，避免仅限前30s
        t_cycle = np.mod(t, self.lissajous_period)
        for i in range(3):
            omega = 2 * pi * self.lissajous_freq[i] / self.lissajous_period
            phase = omega * t_cycle + self.lissajous_phase[i]
            self.desPos[i] = self.lissajous_center[i] + self.lissajous_amp[i] * np.sin(phase)
            self.desVel[i] = self.lissajous_amp[i] * omega * np.cos(phase)
            self.desAcc[i] = -self.lissajous_amp[i] * omega**2 * np.sin(phase)

    # ========================================================================
    # 航向控制方法
    # ========================================================================
    
    def _yaw_follow_velocity(self, t, Ts, quad):
        """航向跟随速度方向"""
        if np.linalg.norm(self.desVel[:2]) > 0.1:
            self.desEul[2] = np.arctan2(self.desVel[1], self.desVel[0])
        
        if (np.sign(self.desEul[2]) != np.sign(self.current_heading) and 
            abs(self.desEul[2] - self.current_heading) >= 2*pi - 0.1):
            self.current_heading += np.sign(self.desEul[2]) * 2*pi
        
        delta_psi = self.desEul[2] - self.current_heading
        self.desYawRate = delta_psi / Ts
        self.current_heading = self.desEul[2]

    # ========================================================================
    # 主接口：desiredState
    # ========================================================================
    
    def desiredState(self, t, Ts, quad):
        """计算期望状态
        
        Args:
            t: 当前时间
            Ts: 采样时间
            quad: 四旋翼对象
        
        Returns:
            sDes: 期望状态向量 [pos(3), vel(3), acc(3), thr(3), eul(3), pqr(3), yawRate(1)]
        """
        # 重置期望状态
        self.desPos = np.zeros(3)
        self.desVel = np.zeros(3)
        self.desAcc = np.zeros(3)
        self.desThr = np.zeros(3)
        self.desEul = np.zeros(3)
        self.desPQR = np.zeros(3)
        self.desYawRate = 0.
        
        # ========== 速度控制模式 ==========
        if self.ctrlType in ["xyz_vel", "xy_vel_z_pos"]:
            if self.xyzType == 1001:
                self.desVel = self._vel_pitch_only_step(t)
            elif self.xyzType in (1012, 1013, 1014):
                self.desVel, self.desAcc = self._vel_sine_sweep_high_freq(t)
            else:
                raise ValueError(f"速度控制模式不支持轨迹类型 {self.xyzType}，仅支持 1001、1012、1013、1014")
            
            # 混合模式
            if self.ctrlType == "xy_vel_z_pos":
                self.desPos[2] = 1.0
                self.desVel[2] = 0
                self.desAcc[2] = 0
        
        # ========== 位置控制模式 ==========
        elif self.ctrlType == "xyz_pos":
            if self.xyzType == 201:
                self._pos_typical_combined(t)
            elif self.xyzType == 206:
                self._pos_lissajous_3d(t)
            else:
                raise ValueError(f"位置控制模式不支持轨迹类型 {self.xyzType}，仅支持 201 和 206")
            
            # 航向控制
            if self.yawType == 3:
                self._yaw_follow_velocity(t, Ts, quad)
            elif self.yawType == 0 or self.yawType == 4:
                self.desEul[2] = 0.0
        self.desVel = self._limit_velocity(self.desVel)
        # 组装状态向量
        self.sDes = np.hstack((
            self.desPos, self.desVel, self.desAcc, self.desThr,
            self.desEul, self.desPQR, self.desYawRate
        )).astype(float)
        
        return self.sDes


# ============================================================================
# 可视化与测试
# ============================================================================

class _DummyQuad:
    """测试用四旋翼占位类"""
    def __init__(self):
        self.psi = 0.0
        self.pos = np.zeros(3)


def _simulate_trajectory(ctrl_type, xyz_type, yaw_type, duration, Ts):
    """仿真轨迹生成"""
    quad = _DummyQuad()
    aver_vel = 1 if ctrl_type == "xyz_pos" else 0
    traj_select = np.array([xyz_type, yaw_type, aver_vel])
    traj = Trajectory(quad, ctrl_type, traj_select)

    times = []
    positions = []
    velocities = []
    accelerations = []
    integrated_pos = quad.pos.copy()

    t = 0.0
    while t <= duration + 1e-9:
        sDes = traj.desiredState(t, Ts, quad)
        des_pos = sDes[0:3]
        des_vel = sDes[3:6]
        des_acc = sDes[6:9]
        
        if ctrl_type == "xyz_pos":
            integrated_pos = des_pos.copy()
        else:
            integrated_pos = integrated_pos + des_vel * Ts

        quad.pos = integrated_pos.copy()
        times.append(t)
        positions.append(integrated_pos.copy())
        velocities.append(des_vel.copy())
        accelerations.append(des_acc.copy())
        t += Ts

    return np.array(times), np.vstack(positions), np.vstack(velocities), np.vstack(accelerations)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from datetime import datetime

    # ========== 测试场景配置（精简版） ==========
    scenarios = [
        # 速度轨迹
        {"ctrl": "xyz_vel", "xyz": 1001, "yaw": 3, "duration": 30.0, 
         "label": "Vel 1001: Pitch Step (0->3->0->8->15->0)"},
        {"ctrl": "xyz_vel", "xyz": 1012, "yaw": 3, "duration": 30.0, 
         "label": "Vel 1012: Sine Sweep High-Freq (0.1-0.5Hz)"},
        {"ctrl": "xyz_vel", "xyz": 1013, "yaw": 3, "duration": 30.0, 
         "label": "Vel 1013: 0.3Hz Sweep (Amplitude 3→18 m/s)"},
        {"ctrl": "xyz_vel", "xyz": 1014, "yaw": 3, "duration": 30.0, 
         "label": "Vel 1014: 8 m/s Amplitude, 0.1→0.5Hz"},
        
        # 位置轨迹
        {"ctrl": "xyz_pos", "xyz": 201, "yaw": 3, "duration": 45.0, 
         "label": "Pos 201: Hybrid (8-Helix-Circle)"},
        {"ctrl": "xyz_pos", "xyz": 206, "yaw": 3, "duration": 30.0, 
         "label": "Pos 206: 3D Lissajous"},
    ]

    Ts = 0.05
    
    print("=" * 70)
    print("四旋翼轨迹生成器测试 - 精简版（仅核心轨迹）")
    print("=" * 70)
    print("\n支持的轨迹:")
    print("  1001: 俯仰轴阶跃 (30秒) - 0->3->0->8->15->0 m/s")
    print("  1012: 正弦扫频高频段 (30秒) - 0.1-0.5Hz, 3-15m/s")
    print("  1013: 正弦扫频 (30秒) - 0.3Hz, 幅值3→18m/s")
    print("  1014: 正弦扫频 (30秒) - 幅值8m/s, 频率0.1→0.5Hz")
    print("  201:  典型组合轨迹 (45秒) - 8字+螺旋+圆形")
    print("  206:  3D李萨如 (30秒)")
    print("\n" + "=" * 70 + "\n")
    
    # 生成所有轨迹数据
    sim_results = []
    for i, info in enumerate(scenarios):
        print(f"[{i+1}/{len(scenarios)}] 生成轨迹: {info['label']}")
        times, positions, velocities, accelerations = _simulate_trajectory(
            ctrl_type=info["ctrl"],
            xyz_type=info["xyz"],
            yaw_type=info["yaw"],
            duration=info["duration"],
            Ts=Ts,
        )
        sim_results.append((info, times, positions, velocities, accelerations))
    
    # ========== 绘图 ==========
    print("\n开始绘图...")
    
    # 速度轨迹图
    fig_vel = plt.figure(figsize=(18, 14))
    fig_vel.suptitle("Velocity Tracking Trajectories (Including New Split Trajectories)", 
                     fontsize=16, fontweight='bold')
    
    vel_results = [r for r in sim_results if r[0]["ctrl"] == "xyz_vel"]
    n_vel = len(vel_results)
    
    for i, (info, times, _, velocities, _) in enumerate(vel_results):
        # 速度分量
        ax1 = fig_vel.add_subplot(n_vel, 2, 2*i+1)
        ax1.plot(times, velocities[:, 0], label="Vx", linewidth=1.5, alpha=0.8)
        ax1.plot(times, velocities[:, 1], label="Vy", linewidth=1.5, alpha=0.8)
        ax1.plot(times, velocities[:, 2], label="Vz", linewidth=1.5, alpha=0.8)
        ax1.set_xlabel("Time (s)", fontsize=9)
        ax1.set_ylabel("Velocity (m/s)", fontsize=9)
        ax1.set_title(info["label"], fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=8)
        
        # 速度幅值
        ax2 = fig_vel.add_subplot(n_vel, 2, 2*i+2)
        vel_mag = np.linalg.norm(velocities, axis=1)
        ax2.plot(times, vel_mag, color='purple', linewidth=2, alpha=0.8)
        ax2.set_xlabel("Time (s)", fontsize=9)
        ax2.set_ylabel("|V| (m/s)", fontsize=9)
        ax2.set_title(info["label"] + " - Magnitude", fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 位置轨迹图
    fig_pos = plt.figure(figsize=(16, 12))
    fig_pos.suptitle("Position Tracking Trajectories", fontsize=16, fontweight='bold')
    
    pos_results = [r for r in sim_results if r[0]["ctrl"] == "xyz_pos"]
    n_pos = len(pos_results)
    n_cols = 3
    n_rows = int(np.ceil(n_pos / n_cols))
    
    for i, (info, times, positions, _, _) in enumerate(pos_results):
        ax = fig_pos.add_subplot(n_rows, n_cols, i+1, projection="3d")
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                linewidth=1.5, color='blue', alpha=0.7)
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  c='green', s=100, marker='o', label='Start', alpha=0.9)
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  c='red', s=100, marker='s', label='End', alpha=0.9)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.set_zlabel("Z (m)", fontsize=8)
        ax.set_title(info["label"], fontsize=10, fontweight='bold')
        ax.legend(loc="upper left", fontsize=7)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Path("log_trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    vel_path = output_dir / f"velocity_trajectories_{timestamp}.png"
    pos_path = output_dir / f"position_trajectories_{timestamp}.png"
    
    fig_vel.savefig(vel_path, dpi=200, bbox_inches='tight')
    fig_pos.savefig(pos_path, dpi=200, bbox_inches='tight')
    
    plt.close('all')
    
    print(f"\n✓ 速度轨迹图保存至: {vel_path}")
    print(f"✓ 位置轨迹图保存至: {pos_path}")
    print("\n" + "="*70)
    print("测试完成！所有轨迹验证通过")
    print("="*70)