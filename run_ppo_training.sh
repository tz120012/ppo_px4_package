#!/bin/bash

# PPO训练启动脚本
# 确保在tianshou_env环境中运行

echo "=== PX4四旋翼PPO训练启动 ==="

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tianshou_env

echo "当前环境: $CONDA_DEFAULT_ENV"
echo "Python路径: $(which python3)"

# 检查依赖
python3 -c "
import torch
import tianshou
import gymnasium
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ Tianshou: {tianshou.__version__}')
print(f'✓ Gymnasium: {gymnasium.__version__}')
"

# 设置工作目录
cd /home/ai/ws_px4_drl/ppo_px4_package

echo "=== 开始快速测试训练（修复后） ==="
python3 ppo_train_px4.py \
    --trajectory_type 5 \
    --yaw_type 3 \
    --max_episode_steps 2000 \
    --dt 0.005 \
    --rl_dt 5.0 \
    --wind_type None \
    --wind_strength 2.0 \
    --seed 0 \
    --buffer_size 256 \
    --hidden_sizes 32 32 \
    --lr 0.0003 \
    --gamma 0.99 \
    --epoch 10 \
    --epoch_num_steps 2000 \
    --collection_step_num_env_steps 128 \
    --update_step_num_repetitions 10 \
    --batch_size 64 \
    --num_train_envs 8 \
    --num_test_envs 4 \
    --return_scaling True \
    --vf_coef 0.25 \
    --ent_coef 0.01 \
    --gae_lambda 0.95 \
    --bound_action_method clip \
    --lr_decay True \
    --max_grad_norm 0.5 \
    --eps_clip 0.2 \
    --logdir log_px4 \
    --render 0.0 \
    --logger tensorboard 

echo "=== 训练完成 ==="