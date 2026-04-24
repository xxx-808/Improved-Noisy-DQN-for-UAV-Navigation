# UAV Grid Navigation with Deep Reinforcement Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

This project provides a 2D grid-world UAV navigation setup with a Gym-style interface. The repository keeps two model implementations: an improved Noisy DQN as the main model and a standard DQN as a baseline for comparison.

---

## Highlights

| Component | Description |
|--|--|
| Environment | `UAVGridWorldEnvironment`: local map + global scalar observations, 5 discrete actions including stay |
| Main Model | [`ImprovedNoisyDQNetwork`](uav_rl/agents/improved_noisy_dqn.py): NoisyNet exploration with residual and LayerNorm-based stabilization |
| Baseline | [`DeepQNetworkAgent`](uav_rl/agents/dqn.py): replay buffer with optional Dueling / Double DQN components |

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- gym 0.26+
- Other dependencies are listed in [`requirements.txt`](requirements.txt)

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` is recommended for editable local development and clean `import uav_rl` behavior.

---

## Quick Start

### Inference with the main model

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

### Load a checkpoint

Make sure the checkpoint matches the architecture (default `state_size=31`, `action_size=5`):

```python
ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")
net = ImprovedNoisyDQNetwork(state_size=31, action_size=5)
net.load_state_dict(ckpt["q_network_state_dict"])
```

You can optionally export to TorchScript in `models/` and use `torch.jit.load` for deployment, or inspect the model graph in [Netron](https://github.com/lutzroeder/netron).

---

## Environment Summary

| Item | Value |
|------|------|
| Default observation size | 31 (`grid_size=15`, `local_map_size=5`) |
| Action space | `Discrete(5)`: up / right / down / left / stay |
| Key parameter | `use_local_map` toggles local map features |

---

## Repository Structure

```text
.
├── uav_rl/
│   ├── environment.py
│   └── agents/
│       ├── dqn.py                  # Baseline DQN
│       └── improved_noisy_dqn.py   # Main model
├── models/                         # Optional exports (e.g., TorchScript)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## License

Add a `LICENSE` file (for example MIT or Apache-2.0) at the repository root before publishing.
