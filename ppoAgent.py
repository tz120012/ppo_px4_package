
#!/usr/bin/env python3
"""
PPO Agent 极简版
功能：接收状态 → 推理动作 → 发送动作
"""
import torch
import torch.nn as nn
import numpy as np
import socket
import struct

# ============ 配置 ============
ACTOR_PATH = "./logs/log_train_results/206/251105-230252/best_policy.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOCAL_IP = '127.0.0.1'
RECV_PORT = 5005  # 接收状态
SEND_PORT = 5006  # 发送动作

OBS_DIM = 56
ACTION_DIM = 20

# ============ 网络定义 ============
class Actor(nn.Module):
    def __init__(self, obs_dim=56, action_dim=20, hidden_sizes=[128, 128]):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        
        self.preprocess = nn.Sequential(*layers)
        self.mu = nn.Linear(hidden_sizes[-1], action_dim)
        self.sigma_param = nn.Parameter(torch.zeros(action_dim, 1))
    
    def forward(self, obs):
        features = self.preprocess(obs)
        mu = self.mu(features)
        sigma = torch.exp(self.sigma_param).squeeze(-1)
        return mu, sigma.expand_as(mu)

# ============ 加载模型 ============
print("正在加载模型...")
checkpoint = torch.load(ACTOR_PATH, map_location=DEVICE)
args = checkpoint.get('args', None) if isinstance(checkpoint, dict) else None
state_dict = (checkpoint.get('model') or checkpoint.get('state_dict') if isinstance(checkpoint, dict) else checkpoint)

if args is not None:
    obs_dim = int(args['state_shape'][0]); act_dim = int(args['action_shape'][0])
    hidden = list(args.get('hidden_sizes', [128, 128]))
else:
    obs_dim, act_dim, hidden = OBS_DIM, ACTION_DIM, [128, 128]
actor = Actor(obs_dim, act_dim, hidden)

# 映射键名
actor_dict = {}
for key, value in state_dict.items():
    if key.startswith('actor.'):
        new_key = key.replace('actor.', '') \
                     .replace('preprocess.model.model.', 'preprocess.') \
                     .replace('mu.model.0.', 'mu.')
        actor_dict[new_key] = value

actor.load_state_dict(actor_dict, strict=False)
actor.to(DEVICE).eval()

print(f"✅ 模型已加载到 {DEVICE}")

# ============ UDP 设置 ============
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind((LOCAL_IP, RECV_PORT))
recv_sock.settimeout(0.01)

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_addr = (LOCAL_IP, SEND_PORT)

print(f"✅ UDP 已初始化 (接收:{RECV_PORT}, 发送:{SEND_PORT})")

# ============ 主循环 ============
print("⏳ 等待状态数据...")

last_obs = None
iteration = 0

try:
    while True:
        # 1. 接收状态
        try:
            data, _ = recv_sock.recvfrom(1024)
            obs = np.array(struct.unpack(f'{len(data)//4}f', data), dtype=np.float32)
            last_obs = obs
            
            if iteration == 0:
                print(f"✅ 开始接收数据 (维度: {len(obs)})")
                print(f"接收到的状态：{obs}")
        except socket.timeout:
            if last_obs is not None:
                obs = last_obs
            else:
                continue
        
        # 2. 推理动作
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action, _ = actor(obs_tensor)
            action = action.squeeze(0).cpu().numpy()
        
        action = np.clip(action, -1.0, 1.0)
        
        # 3. 发送动作
        data = struct.pack(f'{len(action)}f', *action)
        # print(f"✅ 发送动作: {action}")
        send_sock.sendto(data, send_addr)
        
        iteration += 1
        
        # 每 100 次打印一次
        if iteration % 100 == 0:
            print(f"运行 {iteration} 次")

except KeyboardInterrupt:
    print("\n✅ 已停止")
finally:
    recv_sock.close()
    send_sock.close()
