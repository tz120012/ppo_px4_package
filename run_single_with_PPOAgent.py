
# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile
import socket
import struct
import threading

from trajectory import Trajectory
from ctrl import Control
from quadFiles.quad import Quadcopter
from utils.windModel import Wind
import utils
import config

# ============ UDP 通信配置 ============
UDP_CONFIG = {
    'enabled': True,               # 是否启用 UDP 通信
    'local_ip': '127.0.0.1',       # 本地 IP
    'send_port': 5005,             # 发送状态到控制脚本的端口
    'recv_port': 5006,             # 接收控制指令的端口
    'timeout': 0.001,              # 接收超时（秒）
    'buffer_size': 1024,           # 接收缓冲区大小
    'state_freq': 200,              # 状态发送频率 (Hz)
    'action_freq': 200,             # 动作接收频率 (Hz)
}

OBS_DIM = 56            # 观测维度
ACTION_DIM = 20         # 动作维度

# 基线参数
BASELINE_PID_PARAMS = np.array([
    1.0, 0.8,                   # pos_P_gains: Pxy, Pz
    6.5, 8.5,                   # vel_P_gains: Pxydot, Pzdot
    10.0, 1.6,                  # vel_I_gains: Ixydot, Izdot
    0.01, 1.0,                  # vel_D_gains: Dxydot, Dzdot
    8.0, 8.0, 2.75,             # att_P_gains: Pphi, Ptheta, Ppsi
    0.5, 0.5, 0.305,            # rate_P_gains: Pp, Pq, Pr
    3.0, 3.0, 5.0,              # rate_I_gains: Ip, Iq, Ir
    0.008, 0.008, 0.005         # rate_D_gains: Dp, Dq, Dr
], dtype=np.float32)


# ==================== PID参数安全范围（物理边界） ====================
MIN_PID_PARAMS = np.array([
    0.0, 0.1,           # pos_P_gains: Pxy, Pz
    1.2, 2.0,           # vel_P_gains: Pxydot, Pzdot
    0.0, 0.2,           # vel_I_gains: Ixydot, Izdot
    0.1, 0.0,           # vel_D_gains: Dxydot, Dzdot
    0.0, 0.0, 0.0,      # att_P_gains: Pphi, Ptheta, Ppsi
    0.01, 0.01, 0.0,    # rate_P_gains: Pp, Pq, Pr
    0.0, 0.0, 0.0,      # rate_I_gains: Ip, Iq, Ir
    0.0, 0.0, 0.0       # rate_D_gains: Dp, Dq, Dr
], dtype=np.float32)

MAX_PID_PARAMS = np.array([
    2.0, 1.5,           # pos_P_gains
    15.0, 15.0,         # vel_P_gains
    60.0, 3.0,          # vel_I_gains
    2.0, 2.0,           # vel_D_gains
    12.0, 12.0, 5.0,    # att_P_gains
    0.5, 0.6, 0.6,      # rate_P_gains
    10.0, 10.0, 10.0,   # rate_I_gains
    0.01, 0.01, 0.01    # rate_D_gains
], dtype=np.float32)

def normalize_pid_parameters(pid_K_vec):
    # 计算范围
    range_vec = MAX_PID_PARAMS - MIN_PID_PARAMS
    # 归一化到 [-1, 1]
    normalized_pid = 2 * (pid_K_vec - MIN_PID_PARAMS) / range_vec - 1
    return normalized_pid

# ============ UDP 通信类 ============
class QuadUDPInterface:
    """四旋翼仿真 UDP 通信接口"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = config['enabled']
        
        if not self.enabled:
            print("⚠️ UDP 通信已禁用")
            return
        
        # 发送状态的 socket
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.control_addr = (config['local_ip'], config['send_port'])
        
        # 接收控制指令的 socket
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.recv_sock.bind((config['local_ip'], config['recv_port']))
        self.recv_sock.settimeout(config['timeout'])
        
        # 最新接收到的动作
        self.latest_action = normalize_pid_parameters(BASELINE_PID_PARAMS)
        self.action_lock = threading.Lock()

        # 最新解析出的原始 PID 参数向量（20维），用于记录
        self.latest_K_array = BASELINE_PID_PARAMS.copy()
        # 统计信息
        self.sent_count = 0
        self.recv_count = 0
        self.timeout_count = 0
        
        print(f"✅ UDP 通信已启动")
        print(f"   发送状态到: {config['local_ip']}:{config['send_port']}")
        print(f"   接收指令从: {config['local_ip']}:{config['recv_port']}")
    
    def build_state_vector(self, quad, ctrl):
        """
        构建状态向量（56 维）
        # 观测维度计算：
        # - 参考状态: pos(3) + vel(3) + eul(3) + omega(3) = 12
        # - 反馈状态: pos(3) + vel(3) + eul(3) + omega(3) = 12
        # - 误差信号: pos_err(3) + vel_err(3) + eul_err(3) + omega_err(3) = 12
        # - PID参数归一化: 20维（当前PID增益值，归一化到[-1,1]）
        # 总计: 12 + 12 + 12 + 20 = 56
        """
        YPR = utils.quatToYPR_ZYX(ctrl.sDesCalc[9:13])
        # print(f"参考欧拉角 (YPR): {YPR}")
        euler_ref = YPR[::-1]

        pos_err = ctrl.sDesCalc[0:3] - quad.pos
        vel_err = ctrl.sDesCalc[3:6] - quad.vel
        eul_err = euler_ref - quad.euler
        omega_err = ctrl.sDesCalc[10:13] - quad.omega

        state = np.concatenate([
            ctrl.sDesCalc[0:3],     # 参考位置 (3)
            ctrl.sDesCalc[3:6],     # 参考速度 (3)
            euler_ref,              # 参考欧拉角 (3)
            ctrl.sDesCalc[10:13],   # 参考角速度 (3)

            quad.pos,               # 位置 (3)
            quad.vel,               # 速度 (3)
            quad.euler,             # 欧拉角 (3)
            quad.omega,             # 角速度 (3)

            pos_err,                # 位置误差 (3)
            vel_err,                # 速度误差 (3)
            eul_err,                # 欧拉角误差 (3)
            omega_err,              # 角速度误差 (3)

            self.latest_action if self.latest_action is not None else np.zeros(20)  # PID 参数归一化 (20)
        ])
        # print(f"当前状态：{state}")
        
        return state.astype(np.float32)
    
    def send_state(self, quad, ctrl):
        """发送无人机状态（二进制格式）"""
        if not self.enabled:
            return
        
        try:
            state = self.build_state_vector(quad, ctrl)
            # print(f"✅ 发送状态: {state}")
            data = struct.pack(f'{len(state)}f', *state)
            self.send_sock.sendto(data, self.control_addr)
            self.sent_count += 1
        except Exception as e:
            print(f"❌ 发送状态失败: {e}")
    
    def receive_action(self):
        """接收控制动作（非阻塞）"""
        if not self.enabled:
            print("⚠️ UDP 通信未启用，无法接收动作")
            return None
        
        try:
            data, addr = self.recv_sock.recvfrom(self.config['buffer_size'])
            # print(f"✅ 接收到来自 {addr} 的数据: {data}")
            num_floats = len(data) // 4
            action = struct.unpack(f'{num_floats}f', data)
            action = np.array(action, dtype=np.float32)
            
            with self.action_lock:
                self.latest_action = action
            
            self.recv_count += 1
            return action
            
        except socket.timeout:
            self.timeout_count += 1
            return None
        except Exception as e:
            print(f"❌ 接收动作失败: {e}")
            return None
    
    def apply_pid_parameters(self, action: np.ndarray) -> dict:
        """
        将动作映射为 PID 参数字典（适配 ctrl.update_gains 格式）
        
        参数:
            action: numpy 数组 (20,)，范围 [-1, 1]
        
        返回:
            字典格式的 PID 参数，键名为：
            - 'pos_P_gains': [Pxy, Pz]  -> 扩展为 [Px, Py, Pz]
            - 'vel_P_gains': [Pxydot, Pzdot] -> 扩展为 [Pxdot, Pydot, Pzdot]
            - 'vel_I_gains': [Ixydot, Izdot] -> 扩展为 [Ixdot, Iydot, Izdot]
            - 'vel_D_gains': [Dxydot, Dzdot] -> 扩展为 [Dxdot, Dydot, Dzdot]
            - 'att_P_gains': [Pphi, Ptheta, Ppsi]
            - 'rate_P_gains': [Pp, Pq, Pr]
            - 'rate_I_gains': [Ip, Iq, Ir]
            - 'rate_D_gains': [Dp, Dq, Dr]
        """
        action = np.asarray(action, dtype=np.float32).ravel()
        if action.shape[0] != ACTION_DIM:
            print(f"❌ 动作维度错误: 期望 {ACTION_DIM}, 实际 {action.shape[0]}")
            return None
        
        # 将动作映射为 PID 参数
        # action ∈ [-1,1] 映射到 PID 参数的安全范围 [MIN_PID_PARAMS, MAX_PID_PARAMS]
        range_vec = MAX_PID_PARAMS - MIN_PID_PARAMS
        K_array = MIN_PID_PARAMS + (action + 1.0) / 2.0 * range_vec

        # 保存最新原始 PID 向量，供外部记录使用
        self.latest_K_array = K_array.copy()
        
        # 构建字典格式（注意：你的控制器期望 xyz 三维，但训练只用了 xy+z 两维）
        gains_dict = {
            # 位置 P 增益：[Pxy, Pz] -> [Px, Py, Pz]
            'pos_P_gains': [K_array[0], K_array[0], K_array[1]],
            
            # 速度 P 增益：[Pxydot, Pzdot] -> [Pxdot, Pydot, Pzdot]
            'vel_P_gains': [K_array[2], K_array[2], K_array[3]],
            
            # 速度 I 增益：[Ixydot, Izdot] -> [Ixdot, Iydot, Izdot]
            'vel_I_gains': [K_array[4], K_array[4], K_array[5]],
            
            # 速度 D 增益：[Dxydot, Dzdot] -> [Dxdot, Dydot, Dzdot]
            'vel_D_gains': [K_array[6], K_array[6], K_array[7]],
            
            # 姿态 P 增益：[Pphi, Ptheta, Ppsi]
            'att_P_gains': [K_array[8], K_array[9], K_array[10]],
            
            # 角速率 P 增益：[Pp, Pq, Pr]
            'rate_P_gains': [K_array[11], K_array[12], K_array[13]],
            
            # 角速率 I 增益：[Ip, Iq, Ir]
            'rate_I_gains': [K_array[14], K_array[15], K_array[16]],
            
            # 角速率 D 增益：[Dp, Dq, Dr]
            'rate_D_gains': [K_array[17], K_array[18], K_array[19]]
        }
        
        return gains_dict
    
    def get_stats(self):
        """获取通信统计"""
        return {
            'sent': self.sent_count,
            'received': self.recv_count,
            'timeouts': self.timeout_count
        }
    
    def close(self):
        """关闭 socket"""
        if self.enabled:
            self.send_sock.close()
            self.recv_sock.close()
            print("✅ UDP 通信已关闭")


# ============ 主仿真函数（带 UDP） ============
def quad_sim(t, Ts, quad, ctrl, wind, traj, udp_interface=None):
    
    # Dynamics (using last timestep's commands)
    # ---------------------------
    quad.update(t, Ts, ctrl.w_cmd, wind)
    t += Ts

    # Trajectory for Desired States 
    # ---------------------------
    sDes = traj.desiredState(t, Ts, quad)        

    # Generate Commands (for next iteration)
    # ---------------------------
    ctrl.controller(traj, quad, sDes, Ts)

    return t


# ============ UDP 通信线程（高频接收） ============
class ActionReceiverThread(threading.Thread):
    """独立线程用于高频接收动作"""
    
    def __init__(self, udp_interface):
        super().__init__(daemon=True)
        self.udp_interface = udp_interface
        self.running = True
    
    def run(self):
        while self.running:
            self.udp_interface.receive_action()
            time.sleep(1.0 / UDP_CONFIG['action_freq'])
    
    def stop(self):
        self.running = False


# ============ 主函数 ============
def main():
    start_time = time.time()

    # Simulation Setup
    # --------------------------- 
    Ti = 0
    Ts = 0.005
    Tf = 30
    ifsave = 0

    # Choose trajectory settings
    # --------------------------- 
    ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]
    trajSelect = np.zeros(3)

    # 选择控制类型 (0: xyz_pos, 1: xy_vel_z_pos, 2: xyz_vel)
    ctrlType = ctrlOptions[0]
    
    # Select Position Trajectory Type
    trajSelect[0] = 206

    # Select Yaw Trajectory Type
    trajSelect[1] = 3
    
    # Select if waypoint time is used
    trajSelect[2] = 1
    
    # print("Control type: {}".format(ctrlType))

    # Initialize Quadcopter, Controller, Wind, Result Matrixes
    # ---------------------------
    quad = Quadcopter(Ti)
    traj = Trajectory(quad, ctrlType, trajSelect)
    ctrl = Control(quad, traj.yawType)
    wind = Wind('None', 2.0, 90, -15)

    # Initialize UDP Interface
    # ---------------------------
    udp_interface = QuadUDPInterface(UDP_CONFIG)

    # 确保 latest_K_array 有初始值
    if not hasattr(udp_interface, 'latest_K_array'):
        udp_interface.latest_K_array = BASELINE_PID_PARAMS.copy()

    # Trajectory for First Desired States
    # ---------------------------
    sDes = traj.desiredState(0, Ts, quad)

    # Generate First Commands
    # ---------------------------
    ctrl.controller(traj, quad, sDes, Ts)
    
    # Initialize Result Matrixes
    # ---------------------------
    numTimeStep = int(Tf/Ts+1)

    # 用于在每个 step 记录 PID 参数（原始 20 维）
    pid_params_all = np.zeros([numTimeStep, BASELINE_PID_PARAMS.shape[0]], dtype=np.float32)
    pid_params_all[0] = udp_interface.latest_K_array.copy()

    t_all          = np.zeros(numTimeStep)
    s_all          = np.zeros([numTimeStep, len(quad.state)])
    pos_all        = np.zeros([numTimeStep, len(quad.pos)])
    vel_all        = np.zeros([numTimeStep, len(quad.vel)])
    quat_all       = np.zeros([numTimeStep, len(quad.quat)])
    omega_all      = np.zeros([numTimeStep, len(quad.omega)])
    euler_all      = np.zeros([numTimeStep, len(quad.euler)])
    sDes_traj_all  = np.zeros([numTimeStep, len(traj.sDes)])
    sDes_calc_all  = np.zeros([numTimeStep, len(ctrl.sDesCalc)])
    w_cmd_all      = np.zeros([numTimeStep, len(ctrl.w_cmd)])
    wMotor_all     = np.zeros([numTimeStep, len(quad.wMotor)])
    thr_all        = np.zeros([numTimeStep, len(quad.thr)])
    tor_all        = np.zeros([numTimeStep, len(quad.tor)])

    t_all[0]            = Ti
    s_all[0,:]          = quad.state
    pos_all[0,:]        = quad.pos
    vel_all[0,:]        = quad.vel
    quat_all[0,:]       = quad.quat
    omega_all[0,:]      = quad.omega
    euler_all[0,:]      = quad.euler
    sDes_traj_all[0,:]  = traj.sDes
    sDes_calc_all[0,:]  = ctrl.sDesCalc
    w_cmd_all[0,:]      = ctrl.w_cmd
    wMotor_all[0,:]     = quad.wMotor
    thr_all[0,:]        = quad.thr
    tor_all[0,:]        = quad.tor

    # 计算 UDP 发送和接收的间隔
    state_send_interval = int((1.0 / UDP_CONFIG['state_freq']) / Ts)
    action_recv_interval = int((1.0 / UDP_CONFIG['action_freq']) / Ts)

    # Run Simulation
    # ---------------------------
    t = Ti
    i = 1
    pid_update_count = 0
    
    try:
        while round(t, 3) < Tf:
            
            # 1. 以指定频率发送状态
            if UDP_CONFIG['enabled'] and i % state_send_interval == 0:
                udp_interface.send_state(quad, ctrl)
            
            # 2. 以指定频率接收动作并更新 PID
            if UDP_CONFIG['enabled'] and i % action_recv_interval == 0:
                action = udp_interface.receive_action()
                
                if action is not None:
                    # 将动作映射为 PID 参数字典
                    pid_params = udp_interface.apply_pid_parameters(action)
                    
                    if pid_params is not None:
                        # 更新控制器 PID 参数
                        if hasattr(ctrl, 'update_gains'):
                            ctrl.update_gains(pid_params)
                            pid_update_count += 1
                            
                            # # 每 1000 次更新打印一次
                            # if pid_update_count % 1000 == 0:
                            #     print(f"🎯 PID 更新 #{pid_update_count}:")
                            #     # print(f"   位置 P: {pid_params['pos_P_gains']}")
                            #     # print(f"   速度 P: {pid_params['vel_P_gains']}")
                            #     # print(f"   姿态 P: {pid_params['att_P_gains']}")
            
            # 3. 仿真步进
            t = quad_sim(t, Ts, quad, ctrl, wind, traj, udp_interface)
            
            # 4. 记录数据
            t_all[i]             = t
            s_all[i,:]           = quad.state
            pos_all[i,:]         = quad.pos
            vel_all[i,:]         = quad.vel
            quat_all[i,:]        = quad.quat
            omega_all[i,:]       = quad.omega
            euler_all[i,:]       = quad.euler
            sDes_traj_all[i,:]   = traj.sDes
            sDes_calc_all[i,:]   = ctrl.sDesCalc
            w_cmd_all[i,:]       = ctrl.w_cmd
            wMotor_all[i,:]      = quad.wMotor
            thr_all[i,:]         = quad.thr
            tor_all[i,:]         = quad.tor
            # 记录当前 step 的 PID 参数向量（若无人更新则为上一次值）
            pid_params_all[i]    = udp_interface.latest_K_array.copy()
            i += 1
            
            # 5. 定期输出统计信息
            if UDP_CONFIG['enabled'] and i % 1000 == 0:
                stats = udp_interface.get_stats()
                print(f"📊 t={t:.2f}s | UDP 状态: 发送={stats['sent']}, 接收={stats['received']}, "
                      f"超时={stats['timeouts']}, PID更新={pid_update_count}")
    
    except KeyboardInterrupt:
        print("\n🛑 仿真被用户中断")
    finally:
        udp_interface.close()
    
    end_time = time.time()
    print("Simulated {:.2f}s in {:.6f}s.".format(t, end_time - start_time))
    
    if UDP_CONFIG['enabled']:
        final_stats = udp_interface.get_stats()
        print(f"📊 UDP 最终统计: 发送={final_stats['sent']}, 接收={final_stats['received']}, "
              f"超时={final_stats['timeouts']}, PID更新={pid_update_count}")

    # View Results
    # ---------------------------
    # utils.makeFigures(quad.params, t_all, pos_all, vel_all, quat_all, omega_all, euler_all, 
    #                   w_cmd_all, wMotor_all, thr_all, tor_all, sDes_traj_all, sDes_calc_all, trajSelect[0])
    
    utils.makeFigures_ext(quad.params, t_all, pos_all, vel_all, quat_all, omega_all, euler_all, 
                    w_cmd_all, wMotor_all, thr_all, tor_all, sDes_traj_all, sDes_calc_all, trajSelect[0], pid_params_all)
    
    ani = utils.sameAxisAnimation(t_all, traj.wps, pos_all, quat_all, sDes_traj_all, Ts, 
                                  quad.params, traj.xyzType, traj.yawType, ifsave)
    # plt.show()


if __name__ == "__main__":
    if (config.orient == "NED" or config.orient == "ENU"):
        main()
        # cProfile.run('main()')
    else:
        raise Exception("{} is not a valid orientation. Verify config.py file.".format(config.orient))
