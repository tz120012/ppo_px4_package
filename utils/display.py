# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import utils

rad2deg = 180.0/pi
deg2rad = pi/180.0
rads2rpm = 60.0/(2.0*pi)
rpm2rads = 2.0*pi/60.0

# Print complete vector or matrices
def fullprint(*args, **kwargs):
    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    print(*args, **kwargs)
    np.set_printoptions(opt)


def makeFigures(params, time, pos_all, vel_all, quat_all, omega_all, euler_all, commands, wMotor_all, thrust, torque, sDes_traj, sDes_calc,trajSelect):
    x    = pos_all[:,0]
    y    = pos_all[:,1]
    z    = pos_all[:,2]
    q0   = quat_all[:,0]
    q1   = quat_all[:,1]
    q2   = quat_all[:,2]
    q3   = quat_all[:,3]
    xdot = vel_all[:,0]
    ydot = vel_all[:,1]
    zdot = vel_all[:,2]
    p    = omega_all[:,0]*rad2deg
    q    = omega_all[:,1]*rad2deg
    r    = omega_all[:,2]*rad2deg

    wM1  = wMotor_all[:,0]*rads2rpm
    wM2  = wMotor_all[:,1]*rads2rpm
    wM3  = wMotor_all[:,2]*rads2rpm
    wM4  = wMotor_all[:,3]*rads2rpm

    phi   = euler_all[:,0]*rad2deg
    theta = euler_all[:,1]*rad2deg
    psi   = euler_all[:,2]*rad2deg

    x_sp  = sDes_calc[:,0]
    y_sp  = sDes_calc[:,1]
    z_sp  = sDes_calc[:,2]
    Vx_sp = sDes_calc[:,3]
    Vy_sp = sDes_calc[:,4]
    Vz_sp = sDes_calc[:,5]
    x_thr_sp = sDes_calc[:,6]
    y_thr_sp = sDes_calc[:,7]
    z_thr_sp = sDes_calc[:,8]
    q0Des = sDes_calc[:,9]
    q1Des = sDes_calc[:,10]
    q2Des = sDes_calc[:,11]
    q3Des = sDes_calc[:,12]    
    pDes  = sDes_calc[:,13]*rad2deg
    qDes  = sDes_calc[:,14]*rad2deg
    rDes  = sDes_calc[:,15]*rad2deg

    x_tr  = sDes_traj[:,0]
    y_tr  = sDes_traj[:,1]
    z_tr  = sDes_traj[:,2]
    Vx_tr = sDes_traj[:,3]
    Vy_tr = sDes_traj[:,4]
    Vz_tr = sDes_traj[:,5]
    Ax_tr = sDes_traj[:,6]
    Ay_tr = sDes_traj[:,7]
    Az_tr = sDes_traj[:,8]
    yaw_tr = sDes_traj[:,14]*rad2deg

    uM1 = commands[:,0]*rads2rpm
    uM2 = commands[:,1]*rads2rpm
    uM3 = commands[:,2]*rads2rpm
    uM4 = commands[:,3]*rads2rpm

    x_err = x_sp - x
    y_err = y_sp - y
    z_err = z_sp - z

    psiDes   = np.zeros(q0Des.shape[0])
    thetaDes = np.zeros(q0Des.shape[0])
    phiDes   = np.zeros(q0Des.shape[0])
    for ii in range(q0Des.shape[0]):
        YPR = utils.quatToYPR_ZYX(sDes_calc[ii,9:13])
        psiDes[ii]   = YPR[0]*rad2deg
        thetaDes[ii] = YPR[1]*rad2deg
        phiDes[ii]   = YPR[2]*rad2deg

    ###################################################################
    import os
    from datetime import datetime
    # 将t_all、pos_all、vel_all、euler_all、omega_all和sDes_calc_all保存在一个.csv文件，方便后续处理，保存目录为./logs/baseline_pid_results，文件名为datatime_baseline_pid.csv
    # Save results to CSV
    # self.sDesCalc[0:3] = self.pos_sp
    # self.sDesCalc[3:6] = self.vel_sp
    # self.sDesCalc[6:9] = self.thrust_sp
    # self.sDesCalc[9:13] = self.qd
    # self.sDesCalc[13:16] = self.rate_sp
    # ---------------------------
    # Create logs directory if it doesn't exist
    traj_str = str(int(trajSelect))
    log_dir = f'./logs/baseline_pid_results/{traj_str}/'
    os.makedirs(log_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{timestamp}_baseline_pid.csv'
    filepath = os.path.join(log_dir, filename)

    # Combine all data into a single array
    data_to_save = np.column_stack([
        time,
        x,y,z,
        xdot,ydot,zdot,
        phi,theta,psi,
        p,q,r,
        x_sp,y_sp,z_sp,
        Vx_sp,Vy_sp,Vz_sp,
        phiDes,thetaDes,psiDes,
        pDes,qDes,rDes,
    ])

    # Create header
    header = 'time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,euler_roll,euler_pitch,euler_yaw,omega_x,omega_y,omega_z,'
    header += 'pos_x_sp,pos_y_sp,pos_z_sp,vel_x_sp,vel_y_sp,vel_z_sp,euler_roll_sp,euler_pitch_sp,euler_yaw_sp,omega_x_sp,omega_y_sp,omega_z_sp,'

    # Save to CSV
    np.savetxt(filepath, data_to_save, delimiter=',', header=header, comments='')
    print(f"Results saved to: {filepath}")
    ###################################################################

    # plt.show()

    plt.figure()
    plt.plot(time, x, time, y, time, z)
    plt.plot(time, x_sp, '--', time, y_sp, '--', time, z_sp, '--')
    plt.grid(True)
    plt.legend(['x','y','z','x_sp','y_sp','z_sp'])
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.draw()
    plt.savefig(os.path.join(log_dir, f'{timestamp}_position.svg'), dpi=300, bbox_inches='tight')

    plt.figure()
    plt.plot(time, xdot, time, ydot, time, zdot)
    plt.plot(time, Vx_sp, '--', time, Vy_sp, '--', time, Vz_sp, '--')
    plt.grid(True)
    plt.legend(['Vx','Vy','Vz','Vx_sp','Vy_sp','Vz_sp'])
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.draw()
    # 保存图像
    plt.savefig(os.path.join(log_dir, f'{timestamp}_velocity.svg'), dpi=300, bbox_inches='tight')

    plt.figure()
    plt.plot(time, x_thr_sp, time, y_thr_sp, time, z_thr_sp)
    plt.grid(True)
    plt.legend(['x_thr_sp','y_thr_sp','z_thr_sp'])
    plt.xlabel('Time (s)')
    plt.ylabel('Desired Thrust (N)')
    # plt.draw()

    plt.figure()
    plt.plot(time, phi, time, theta, time, psi)
    plt.plot(time, phiDes, '--', time, thetaDes, '--', time, psiDes, '--')
    plt.plot(time, yaw_tr, '-.')
    plt.grid(True)
    plt.legend(['roll','pitch','yaw','roll_sp','pitch_sp','yaw_sp','yaw_tr'])
    plt.xlabel('Time (s)')
    plt.ylabel('Euler Angle (°)')
    plt.draw()
    # 保存图像
    plt.savefig(os.path.join(log_dir, f'{timestamp}_attitude.svg'), dpi=300, bbox_inches='tight')

    plt.figure()
    plt.plot(time, p, time, q, time, r)
    plt.plot(time, pDes, '--', time, qDes, '--', time, rDes, '--')
    plt.grid(True)
    plt.legend(['p','q','r','p_sp','q_sp','r_sp'])
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (°/s)')
    plt.draw()
    # 保存图像
    plt.savefig(os.path.join(log_dir, f'{timestamp}_angular_velocity.svg'), dpi=300, bbox_inches='tight')

    plt.figure()
    plt.plot(time, wM1, time, wM2, time, wM3, time, wM4)
    plt.plot(time, uM1, '--', time, uM2, '--', time, uM3, '--', time, uM4, '--')
    plt.grid(True)
    plt.legend(['w1','w2','w3','w4'])
    plt.xlabel('Time (s)')
    plt.ylabel('Motor Angular Velocity (RPM)')
    # plt.draw()

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time, thrust[:,0], time, thrust[:,1], time, thrust[:,2], time, thrust[:,3])
    plt.grid(True)
    plt.legend(['thr1','thr2','thr3','thr4'], loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotor Thrust (N)')
    # plt.draw()

    plt.subplot(2,1,2)
    plt.plot(time, torque[:,0], time, torque[:,1], time, torque[:,2], time, torque[:,3])
    plt.grid(True)
    plt.legend(['tor1','tor2','tor3','tor4'], loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotor Torque (N*m)')
    # plt.draw()

    plt.figure()
    plt.subplot(3,1,1)
    plt.title('Trajectory Setpoints')
    plt.plot(time, x_tr, time, y_tr, time, z_tr)
    plt.grid(True)
    plt.legend(['x','y','z'], loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')

    plt.subplot(3,1,2)
    plt.plot(time, Vx_tr, time, Vy_tr, time, Vz_tr)
    plt.grid(True)
    plt.legend(['Vx','Vy','Vz'], loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
   
    plt.subplot(3,1,3)
    plt.plot(time, Ax_tr, time, Ay_tr, time, Az_tr)
    plt.grid(True)
    plt.legend(['Ax','Ay','Az'], loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    # plt.draw()

    plt.figure()
    plt.plot(time, x_err, time, y_err, time, z_err)
    plt.grid(True)
    plt.legend(['Pos x error','Pos y error','Pos z error'])
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error (m)')
    # plt.draw()