import os
import sys
import numpy as np

# ensure project root is on sys.path so tests can import px4Env
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from px4Env import QuadcopterEnv


def test_freeze_gating_behavior():
    env = QuadcopterEnv(error_gating_enabled=True)
    # set prev_action to zeros, action to ones
    env.prev_action = np.zeros(20, dtype=np.float32)
    action = np.ones(20, dtype=np.float32)
    env.current_pid_domain = 'low'

    # very small errors -> should freeze groups
    errors = np.zeros(12, dtype=np.float32)
    gated = env._error_gated_action_adaptation(action, errors)

    # pos_P (0:2) and vel_P (2:4) and other groups should be frozen (equal to prev_action)
    assert np.allclose(gated[0:2], env.prev_action[0:2])
    assert np.allclose(gated[2:4], env.prev_action[2:4])


def test_limited_gating_behavior():
    env = QuadcopterEnv(error_gating_enabled=True)
    env.prev_action = np.zeros(20, dtype=np.float32)
    action = np.ones(20, dtype=np.float32)
    env.current_pid_domain = 'medium'

    # choose vel_err so that error_ratio == 0.5 for medium domain
    t1 = 0.15
    t2 = 0.5
    vel_err_val = (t1 + t2) / 2.0  # 0.325
    errors = np.zeros(12, dtype=np.float32)
    errors[3] = vel_err_val

    gated = env._error_gated_action_adaptation(action, errors)

    # for vel_P group (indices 2:4), expected limit_delta = 0.1 + 0.4*0.5 = 0.3
    expected = np.full(2, 0.3, dtype=np.float32)
    assert np.allclose(gated[2:4], expected, atol=1e-6)


def test_free_gating_behavior():
    env = QuadcopterEnv(error_gating_enabled=True)
    env.prev_action = np.zeros(20, dtype=np.float32)
    action = np.ones(20, dtype=np.float32)
    env.current_pid_domain = 'high'

    # set attitude error large -> should be free for att_P group (8:11)
    errors = np.zeros(12, dtype=np.float32)
    errors[6] = 1.0

    gated = env._error_gated_action_adaptation(action, errors)

    assert np.allclose(gated[8:11], action[8:11])


def test_apply_pid_parameters_with_gating_returns_gains_dict():
    env = QuadcopterEnv(error_gating_enabled=True)
    env.prev_action = np.zeros(20, dtype=np.float32)
    action = np.zeros(20, dtype=np.float32)
    errors = np.zeros(12, dtype=np.float32)

    gains = env.apply_pid_parameters_with_gating(action, errors)
    assert isinstance(gains, dict)
    # basic keys expected
    expected_keys = {
        'pos_P_gains', 'vel_P_gains', 'vel_I_gains', 'vel_D_gains',
        'att_P_gains', 'rate_P_gains', 'rate_I_gains', 'rate_D_gains'
    }
    assert expected_keys.issubset(set(gains.keys()))
