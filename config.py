# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

# Select Orientation of Quadcopter and Reference Frame
# ---------------------------
# "NED" for front-right-down (frd) and North-East-Down
# "ENU" for front-left-up (flu) and East-North-Up
orient = "NED"

# Select whether to use gyroscopic precession of the rotors in the quadcopter dynamics
# ---------------------------
# Set to False if rotor inertia isn't known (gyro precession has negigeable effect on drone dynamics)
usePrecession = bool(False)

## ########## 扫幅+扫频参数 ############
# -----------------------------
# Rate 模式（角速度）扫频 + 扫幅
# -----------------------------
RATE_SWEEP_LOW_HZ      = 0.0
RATE_SWEEP_HIGH_HZ     = 1.0
RATE_SWEEP_PERIOD_S    = 30.0
RATE_SWEEP_MODE        = "lin"   # "lin" or "log"

RATE_AMP_P_LOW_DEGPS   = 0.0     # deg/s
RATE_AMP_P_HIGH_DEGPS  = 100.0
RATE_AMP_Q_LOW_DEGPS   = 0.0
RATE_AMP_Q_HIGH_DEGPS  = 100.0
RATE_AMP_R_LOW_DEGPS   = 0.0
RATE_AMP_R_HIGH_DEGPS  = 50.0
RATE_AMP_PERIOD_S      = 30.0    # 扫幅周期（可与扫频不同）

# -----------------------------
# Att 模式（姿态角）扫频 + 扫幅
# -----------------------------
ATT_SWEEP_LOW_HZ       = 0.0
ATT_SWEEP_HIGH_HZ      = 15.0
ATT_SWEEP_PERIOD_S     = 30.0
ATT_SWEEP_MODE         = "lin"

ATT_AMP_ROLL_LOW_DEG   = 0.0     # deg
ATT_AMP_ROLL_HIGH_DEG  = 45.0
ATT_AMP_PITCH_LOW_DEG  = 0.0
ATT_AMP_PITCH_HIGH_DEG = 45.0
ATT_AMP_YAW_LOW_DEG    = 0.0
ATT_AMP_YAW_HIGH_DEG   = 30.0
ATT_AMP_PERIOD_S       = 30.0

# -----------------------------
# Vel 模式（速度）扫频 + 扫幅
# -----------------------------
VEL_SWEEP_LOW_HZ       = 0.0
VEL_SWEEP_HIGH_HZ      = 3.0
VEL_SWEEP_PERIOD_S     = 60.0
VEL_SWEEP_MODE         = "lin"

VEL_AMP_VX_LOW_MS      = 0.0     # m/s
VEL_AMP_VX_HIGH_MS     = 15.0
VEL_AMP_VY_LOW_MS      = 0.0
VEL_AMP_VY_HIGH_MS     = 15.0
VEL_AMP_VZ_LOW_MS      = 0.0     # 定高 => 保持 0
VEL_AMP_VZ_HIGH_MS     = 0.0
VEL_AMP_PERIOD_S       = 60.0

# 可选：统一相位偏移
ALT_SWEEP_PHASE        = 0.0
