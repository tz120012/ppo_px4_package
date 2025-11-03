## 终端开启屏幕记录
clear
tmux new -s rl && tmux set -g mouse on

conda activate tianshou_env
cd /root/autodl-tmp/ppo_px4_package


## 训练命令
conda activate tianshou_env
python train.py --lr-decay --lr-decay-type linear  --min-lr-ratio 0.05 --grad-norm-decay --grad-norm-decay-type linear --max-grad-norm 1.0 --final-grad-norm 0.01


## 评估命令
conda activate tianshou_env

# 部署后的运行命令

# tensorboard 可视化 命令
conda activate tianshou_env
tensorboard --logdir=logs/log_train_results