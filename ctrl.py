# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

# Position and Velocity Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_pos_control/PositionControl.cpp
# Desired Thrust to Desired Attitude based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_pos_control/Utility/ControlMath.cpp
# Attitude Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_att_control/AttitudeControl/AttitudeControl.cpp
# and https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/154099/eth-7387-01.pdf
# Rate Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_att_control/mc_att_control_main.cpp

import numpy as np
from numpy import pi
from numpy import sin, cos, tan, sqrt
from numpy.linalg import norm
import utils
import config

rad2deg = 180.0/pi
deg2rad = pi/180.0

class Control:
    
    def __init__(self, quad, yawType):
        # Set PID Gains and Max Values
        self.pos_P_gain = np.array([1.0, 1.0, 0.8])
        self.vel_P_gain = np.array([3.1, 3.1, 8.5])
        self.vel_I_gain = np.array([30, 30, 1.6])
        self.vel_D_gain = np.array([1.05, 1.05, 1.0])
        self.att_P_gain = np.array([8, 8, 2.75])
        self.rate_P_gain = np.array([0.255, 0.305, 0.305])
        self.rate_I_gain = np.array([5.0, 5.0, 5.0])
        self.rate_D_gain = np.array([0.005, 0.005, 0.005])

        # Attitude P gains
        self.PpsiStrong = 8

        # Max Velocities
        uMax = 15.0
        vMax = 15.0
        wMax = 5.0

        self.velMax = np.array([uMax, vMax, wMax])
        self.velMaxAll = 20.0

        self.saturateVel_separetely = False

        # Max tilt
        self.tiltMax = 60.0*deg2rad

        # Max Rate
        pMax = 300.0*deg2rad
        qMax = 300.0*deg2rad
        rMax = 150.0*deg2rad

        self.rateMax = np.array([pMax, qMax, rMax])

        # 
        self.sDesCalc = np.zeros(16)
        self.w_cmd = np.ones(4)*quad.params["w_hover"]
        self.thr_int = np.zeros(3)
        if (yawType == 0):
            self.att_P_gain[2] = 0
        self.setYawWeight()
        self.pos_sp    = np.zeros(3)
        self.vel_sp    = np.zeros(3)
        self.acc_sp    = np.zeros(3)
        self.thrust_sp = np.zeros(3)
        self.eul_sp    = np.zeros(3)
        self.pqr_sp    = np.zeros(3)
        self.yawFF     = np.zeros(3)

        self.rate_int = np.zeros(3)
        # 积分器限制值
        self._lim_int = np.array([0.3, 0.3, 0.1])  # [roll, pitch, yaw] 积分限制
        # 着陆状态标志
        self.island = True    

    
    def controller(self, traj, quad, sDes, Ts):

        # Desired State (Create a copy, hence the [:])
        # ---------------------------
        self.pos_sp[:]    = traj.sDes[0:3]
        self.vel_sp[:]    = traj.sDes[3:6]
        self.acc_sp[:]    = traj.sDes[6:9]
        self.thrust_sp[:] = traj.sDes[9:12]
        self.eul_sp[:]    = traj.sDes[12:15]
        self.pqr_sp[:]    = traj.sDes[15:18]
        self.yawFF[:]     = traj.sDes[18]
        
        # 是否为落地状态   
        self.island = quad.landed 

        # Select Controller
        # ---------------------------
        if (traj.ctrlType == "xyz_vel"):
            # print("Using xyz_vel controller")
            self.saturateVel()
            self.z_vel_control(quad, Ts)
            self.xy_vel_control(quad, Ts)
            self.thrustToAttitude(quad, Ts)
            self.attitude_control(quad, Ts)
            self.rate_control(quad, Ts)
        elif (traj.ctrlType == "xy_vel_z_pos"):
            self.z_pos_control(quad, Ts)
            self.saturateVel()
            self.z_vel_control(quad, Ts)
            self.xy_vel_control(quad, Ts)
            self.thrustToAttitude(quad, Ts)
            self.attitude_control(quad, Ts)
            self.rate_control(quad, Ts)
        elif (traj.ctrlType == "xyz_pos"):
            self.z_pos_control(quad, Ts)
            self.xy_pos_control(quad, Ts)
            self.saturateVel()
            self.z_vel_control(quad, Ts)
            self.xy_vel_control(quad, Ts)
            self.thrustToAttitude(quad, Ts)
            self.attitude_control(quad, Ts)
            self.rate_control(quad, Ts)

        # Mixer
        # --------------------------- 
        self.w_cmd = utils.mixerFM(quad, norm(self.thrust_sp), self.rateCtrl)
        
        # Add calculated Desired States
        # ---------------------------         
        self.sDesCalc[0:3] = self.pos_sp
        self.sDesCalc[3:6] = self.vel_sp
        self.sDesCalc[6:9] = self.thrust_sp
        self.sDesCalc[9:13] = self.qd
        self.sDesCalc[13:16] = self.rate_sp


    def z_pos_control(self, quad, Ts):
       
        # Z Position Control
        # --------------------------- 
        pos_z_error = self.pos_sp[2] - quad.pos[2]
        self.vel_sp[2] += self.pos_P_gain[2]*pos_z_error
        
    
    def xy_pos_control(self, quad, Ts):

        # XY Position Control
        # --------------------------- 
        pos_xy_error = (self.pos_sp[0:2] - quad.pos[0:2])
        self.vel_sp[0:2] += self.pos_P_gain[0:2]*pos_xy_error
        
        
    def saturateVel(self):

        # Saturate Velocity Setpoint
        # --------------------------- 
        # Either saturate each velocity axis separately, or total velocity (prefered)
        if (self.saturateVel_separetely):
            self.vel_sp = np.clip(self.vel_sp, -self.velMax, self.velMax)
        else:
            totalVel_sp = norm(self.vel_sp)
            if (totalVel_sp > self.velMaxAll):
                self.vel_sp = self.vel_sp/totalVel_sp*self.velMaxAll


    def z_vel_control(self, quad, Ts):
        
        # Z Velocity Control (Thrust in D-direction)
        # ---------------------------
        # Hover thrust (m*g) is sent as a Feed-Forward term, in order to 
        # allow hover when the position and velocity error are nul
        vel_z_error = self.vel_sp[2] - quad.vel[2]
        
        # Initialize thrust_z_sp with default calculation
        base_thrust = self.vel_P_gain[2]*vel_z_error - self.vel_D_gain[2]*quad.vel_dot[2] + quad.params["mB"]*self.acc_sp[2] + self.thr_int[2]
        
        if (config.orient == "NED"):
            thrust_z_sp = base_thrust - quad.params["mB"]*quad.params["g"]
        elif (config.orient == "ENU"):
            thrust_z_sp = base_thrust + quad.params["mB"]*quad.params["g"]
        else:
            # Default to ENU orientation if config.orient is not recognized
            thrust_z_sp = base_thrust + quad.params["mB"]*quad.params["g"]
        
        # Get thrust limits
        if (config.orient == "NED"):
            # The Thrust limits are negated and swapped due to NED-frame
            uMax = -quad.params["minThr"]
            uMin = -quad.params["maxThr"]
        elif (config.orient == "ENU"):
            uMax = quad.params["maxThr"]
            uMin = quad.params["minThr"]
        else:
            # Default to NED orientation if config.orient is not recognized
            uMax = -quad.params["minThr"]
            uMin = -quad.params["maxThr"]

        # Apply Anti-Windup in D-direction
        stop_int_D = (thrust_z_sp >= uMax and vel_z_error >= 0.0) or (thrust_z_sp <= uMin and vel_z_error <= 0.0)

        # Calculate integral part
        if not (stop_int_D):
            self.thr_int[2] += self.vel_I_gain[2]*vel_z_error*Ts * quad.params["useIntergral"]
            # Limit thrust integral
            self.thr_int[2] = min(abs(self.thr_int[2]), quad.params["maxThr"])*np.sign(self.thr_int[2])

        # Saturate thrust setpoint in D-direction
        self.thrust_sp[2] = np.clip(thrust_z_sp, uMin, uMax)

    
    def xy_vel_control(self, quad, Ts):
        
        # XY Velocity Control (Thrust in NE-direction)
        # ---------------------------
        vel_xy_error = self.vel_sp[0:2] - quad.vel[0:2]
        thrust_xy_sp = self.vel_P_gain[0:2]*vel_xy_error - self.vel_D_gain[0:2]*quad.vel_dot[0:2] + quad.params["mB"]*(self.acc_sp[0:2]) + self.thr_int[0:2]

        # Max allowed thrust in NE based on tilt and excess thrust
        thrust_max_xy_tilt = abs(self.thrust_sp[2])*np.tan(self.tiltMax)
        thrust_max_xy = sqrt(quad.params["maxThr"]**2 - self.thrust_sp[2]**2)
        thrust_max_xy = min(thrust_max_xy, thrust_max_xy_tilt)

        # Saturate thrust in NE-direction
        self.thrust_sp[0:2] = thrust_xy_sp
        if (np.dot(self.thrust_sp[0:2].T, self.thrust_sp[0:2]) > thrust_max_xy**2):
            mag = norm(self.thrust_sp[0:2])
            self.thrust_sp[0:2] = thrust_xy_sp/mag*thrust_max_xy
        
        # Use tracking Anti-Windup for NE-direction: during saturation, the integrator is used to unsaturate the output
        # see Anti-Reset Windup for PID controllers, L.Rundqwist, 1990
        # 添加数值稳定性检查，防止除零错误
        vel_P_safe = np.maximum(self.vel_P_gain[0:2], 1e-6)  # 确保最小值为1e-6
        arw_gain = 2.0/vel_P_safe
        vel_err_lim = vel_xy_error - (thrust_xy_sp - self.thrust_sp[0:2])*arw_gain
        self.thr_int[0:2] += self.vel_I_gain[0:2]*vel_err_lim*Ts * quad.params["useIntergral"]
    
    def thrustToAttitude(self, quad, Ts):
        
        # Create Full Desired Quaternion Based on Thrust Setpoint and Desired Yaw Angle
        # ---------------------------
        yaw_sp = self.eul_sp[2]

        # Desired body_z axis direction
        body_z = -utils.vectNormalize(self.thrust_sp)
        if (config.orient == "ENU"):
            body_z = -body_z
        
        # Vector of desired Yaw direction in XY plane, rotated by pi/2 (fake body_y axis)
        y_C = np.array([-sin(yaw_sp), cos(yaw_sp), 0.0])
        
        # Desired body_x axis direction
        body_x = np.cross(y_C, body_z)
        body_x = utils.vectNormalize(body_x)
        
        # Desired body_y axis direction
        body_y = np.cross(body_z, body_x)

        # Desired rotation matrix
        R_sp = np.array([body_x, body_y, body_z]).T

        # Full desired quaternion (full because it considers the desired Yaw angle)
        self.qd_full = utils.RotToQuat(R_sp)
        
        
    def attitude_control(self, quad, Ts):

        # Current thrust orientation e_z and desired thrust orientation e_z_d
        e_z = quad.dcm[:,2]
        e_z_d = -utils.vectNormalize(self.thrust_sp)
        if (config.orient == "ENU"):
            e_z_d = -e_z_d

        # Quaternion error between the 2 vectors
        qe_red = np.zeros(4)
        qe_red[0] = np.dot(e_z, e_z_d) + sqrt(norm(e_z)**2 * norm(e_z_d)**2)
        qe_red[1:4] = np.cross(e_z, e_z_d)
        qe_red = utils.vectNormalize(qe_red)
        
        # Reduced desired quaternion (reduced because it doesn't consider the desired Yaw angle)
        self.qd_red = utils.quatMultiply(qe_red, quad.quat)

        # Mixed desired quaternion (between reduced and full) and resulting desired quaternion qd
        q_mix = utils.quatMultiply(utils.inverse(self.qd_red), self.qd_full)
        q_mix = q_mix*np.sign(q_mix[0])
        q_mix[0] = np.clip(q_mix[0], -1.0, 1.0)
        q_mix[3] = np.clip(q_mix[3], -1.0, 1.0)
        self.qd = utils.quatMultiply(self.qd_red, np.array([cos(self.yaw_w*np.arccos(q_mix[0])), 0, 0, sin(self.yaw_w*np.arcsin(q_mix[3]))]))

        # Resulting error quaternion
        self.qe = utils.quatMultiply(utils.inverse(quad.quat), self.qd)

        # Create rate setpoint from quaternion error
        self.rate_sp = (2.0*np.sign(self.qe[0])*self.qe[1:4])*self.att_P_gain
        
        # Limit yawFF
        self.yawFF = np.clip(self.yawFF, -self.rateMax[2], self.rateMax[2])

        # Add Yaw rate feed-forward
        self.rate_sp += utils.quat2Dcm(utils.inverse(quad.quat))[:,2]*self.yawFF

        # Limit rate setpoint
        self.rate_sp = np.clip(self.rate_sp, -self.rateMax, self.rateMax)


    def rate_control(self, quad, Ts):
        
        # Rate Control
        # ---------------------------
        rate_error = self.rate_sp - quad.omega
        self.rateCtrl = self.rate_P_gain*rate_error + self.rate_int- self.rate_D_gain*quad.omega_dot     # Be sure it is right sign for the D part
        
        if not self.island:
            self.update_rate_integral(rate_error,dt=Ts)

    def update_rate_integral(self, rate_error: np.ndarray, dt: float):

        rate_error = np.array(rate_error, dtype=float, copy=True)

        for i in range(3):
            # 防止正向饱和继续积累（将误差正半轴抑制至 <= 0）
            rate_error[i] = min(rate_error[i], 0.0)

            # 防止负向饱和继续积累（将误差负半轴提升至 >= 0）
            rate_error[i] = max(rate_error[i], 0.0)

            # I 项因子：误差越大，I 增益越小（平滑抑制快速积累）
            i_factor = rate_error[i] / 400.0*deg2rad
            i_factor = max(0.0, 1.0 - i_factor * i_factor)

            # 一阶积分
            rate_i = self.rate_int[i] + i_factor * self.rate_I_gain[i] * rate_error[i] * dt

            # 仅当有效时更新，并进行限幅
            if np.isfinite(rate_i):
                self.rate_int[i] = np.clip(rate_i, -self._lim_int[i], self._lim_int[i])


    def setYawWeight(self):
        
        # Calculate weight of the Yaw control gain
        # 添加数值稳定性检查，防止除零错误
        roll_pitch_gain = 0.5*(self.att_P_gain[0] + self.att_P_gain[1])
        
        # 确保 roll_pitch_gain 不为零，防止除零错误
        if roll_pitch_gain < 1e-6:
            roll_pitch_gain = 1e-6
            self.yaw_w = 1.0  # 默认值
        else:
            self.yaw_w = np.clip(self.att_P_gain[2]/roll_pitch_gain, 0.0, 1.0)

        self.att_P_gain[2] = roll_pitch_gain

    def update_gains(self, gains_from_action):
        """更新所有PID增益参数
        
        参数:
            gains_from_action: 包含PID增益的字典
                - 'pos_P_gains': 位置P增益 [Px, Py, Pz]
                - 'vel_P_gains': 速度P增益 [Pxdot, Pydot, Pzdot]
                - 'vel_D_gains': 速度D增益 [Dxdot, Dydot, Dzdot]
                - 'vel_I_gains': 速度I增益 [Ixdot, Iydot, Izdot]
                - 'att_P_gains': 姿态P增益 [Pphi, Ptheta, Ppsi]
                - 'rate_P_gains': 角速率P增益 [Pp, Pq, Pr]
                - 'rate_I_gains': 角速率I增益 [Ip, Iq, Ir]
                - 'rate_D_gains': 角速率D增益 [Dp, Dq, Dr]
        """
        # 定义最小值以防止数值不稳定
        min_gain = 1e-6
        
        # 更新位置P增益（确保最小值）
        if 'pos_P_gains' in gains_from_action:
            self.pos_P_gain = np.maximum(np.array(gains_from_action['pos_P_gains']), min_gain)
            
        # 更新速度PID增益（确保最小值）
        if 'vel_P_gains' in gains_from_action:
            self.vel_P_gain = np.maximum(np.array(gains_from_action['vel_P_gains']), min_gain)
        if 'vel_D_gains' in gains_from_action:
            self.vel_D_gain = np.maximum(np.array(gains_from_action['vel_D_gains']), 0.0)  # D增益可以为0
        if 'vel_I_gains' in gains_from_action:
            self.vel_I_gain = np.maximum(np.array(gains_from_action['vel_I_gains']), 0.0)  # I增益可以为0
            
        # 更新姿态P增益（确保最小值）
        if 'att_P_gains' in gains_from_action:
            self.att_P_gain = np.maximum(np.array(gains_from_action['att_P_gains']), min_gain)
            
        # 更新角速率PID增益
        if 'rate_P_gains' in gains_from_action:
            self.rate_P_gain = np.maximum(np.array(gains_from_action['rate_P_gains']), min_gain)
        if 'rate_I_gains' in gains_from_action:
            self.rate_I_gain = np.maximum(np.array(gains_from_action['rate_I_gains']), 0.0)  # I增益可以为0
        if 'rate_D_gains' in gains_from_action:
            self.rate_D_gain = np.maximum(np.array(gains_from_action['rate_D_gains']), 0.0)  # D增益可以为0
            
        # 更新偏航权重（基于新的姿态增益）
        self.setYawWeight()
