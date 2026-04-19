# UAV 栅格导航 · 深度强化学习

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

基于 **OpenAI Gym** 接口的二维栅格环境：无人机从起点导航至目标，避开障碍并穿越弱信号区域。仓库提供 **改进型 Noisy DQN（主模型）** 与 **标准 DQN 基线**，便于对比实验。

---

## 功能概览

| | |
|--|--|
| **环境** | `UAVGridWorldEnvironment`：局部地图 + 全局标量观测，离散 5 动作（含停留） |
| **主模型** | [`ImprovedNoisyDQNetwork`](uav_rl/agents/improved_noisy_dqn.py) · NoisyNet + 残差 / LayerNorm 等 |
| **基线** | [`DeepQNetworkAgent`](uav_rl/agents/dqn.py) · 经验回放，可选 Dueling / Double DQN |

---

## 环境要求

- **Python** 3.9+
- **PyTorch** 2.0+
- **gym** 0.26+
- 其余见 [`requirements.txt`](requirements.txt)

---

## 安装

```bash
git clone https://github.com/<你的用户名>/<仓库名>.git
cd <仓库名>

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
pip install -e .
```

> `pip install -e .` 为可编辑安装，便于在项目外执行 `import uav_rl`。将 clone 地址换成你的真实仓库 URL 即可。

---

## 快速开始

### 主模型推理

```python
import torch
from uav_rl import UAVGridWorldEnvironment, ImprovedNoisyDQNetwork

env = UAVGridWorldEnvironment()
state = env.reset()

net = ImprovedNoisyDQNetwork(
    state_size=state.shape[0],
    action_size=env.action_space.n,
)
net.eval()

done = False
while not done:
    with torch.no_grad():
        q = net(torch.as_tensor(state, dtype=torch.float32))
    action = int(q.argmax().item())
    state, reward, done, info = env.step(action)
```

### 加载检查点

权重需与网络结构一致（默认 `state_size=31`，`action_size=5`）：

```python
ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")
net = ImprovedNoisyDQNetwork(state_size=31, action_size=5)
net.load_state_dict(ckpt["q_network_state_dict"])
```

可选：将策略导出为 TorchScript 置于 `models/`，使用 `torch.jit.load` 部署，或用 [Netron](https://github.com/lutzroeder/netron) 查看计算图。

---

## 环境说明

| 项目 | 说明 |
|------|------|
| 默认观测维度 | 31（`grid_size=15`，`local_map_size=5`） |
| 动作空间 | `Discrete(5)`：上 / 右 / 下 / 左 / 停留 |
| 参数 | `use_local_map` 控制是否使用局部栅格特征 |

---

## 目录结构

```
.
├── uav_rl/
│   ├── environment.py
│   └── agents/
│       ├── dqn.py                  # 基线 DQN
│       └── improved_noisy_dqn.py   # 主模型
├── models/                         # 可选：TorchScript 等导出
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 开源协议

在仓库根目录添加 `LICENSE`（如 MIT、Apache-2.0）以明确条款；课程或竞赛提交请遵守主办方要求。
