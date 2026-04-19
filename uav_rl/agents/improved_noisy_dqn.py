import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import math

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NoisyLinear(nn.Module):
    """改进的噪声线性层 - 支持自适应噪声调度"""
    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 可学习参数
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # 噪声缓冲区
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        # 噪声强度缩放因子
        self.noise_scale = 1.0
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """初始化参数"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        """缩放噪声"""
        x = torch.randn(size, device=device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def set_noise_scale(self, scale):
        """设置噪声缩放因子"""
        self.noise_scale = scale
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon * self.noise_scale
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon * self.noise_scale
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

class ImprovedNoisyDQNetwork(nn.Module):
    """改进的NoisyNet DQN网络 - 稳定化设计"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ImprovedNoisyDQNetwork, self).__init__()
        
        # 特征提取层 - 使用更稳定的架构
        self.features = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),  # 使用LayerNorm替代BatchNorm
            nn.ReLU(),
            nn.Dropout(0.1),  # 降低dropout率
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 噪声层
        self.noisy1 = NoisyLinear(hidden_size, hidden_size, std_init=0.1)
        self.noisy2 = NoisyLinear(hidden_size, action_size, std_init=0.1)
        
        # 残差连接
        self.residual = nn.Linear(hidden_size, hidden_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 特征提取
        features = self.features(x)
        
        # 噪声层 + 残差连接
        noisy_out = F.relu(self.noisy1(features))
        noisy_out = noisy_out + self.residual(features)  # 残差连接
        
        # 输出层
        q_values = self.noisy2(noisy_out)
        
        return q_values
    
    def reset_noise(self):
        """重置所有噪声层"""
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()
    
    def set_noise_scale(self, scale):
        """设置噪声缩放因子"""
        self.noisy1.set_noise_scale(scale)
        self.noisy2.set_noise_scale(scale)

class ImprovedNoisyDQNAgent:
    """改进的NoisyNet DQN智能体 - 避免暂时性暴跌"""
    def __init__(self, state_size, action_size, hidden_size=128, buffer_size=50000, 
                 batch_size=64, gamma=0.99, lr=3e-4, update_target_every=1000,
                 tau=0.005, use_soft_update=True):
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_target_every = update_target_every
        self.tau = tau
        self.use_soft_update = use_soft_update
        
        # 创建网络
        self.q_network = ImprovedNoisyDQNetwork(state_size, action_size, hidden_size).to(device)
        self.target_network = ImprovedNoisyDQNetwork(state_size, action_size, hidden_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器 - 使用更温和的权重衰减
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-4)
        
        # 学习率调度器 - 预热 + 余弦退火
        self.warmup_steps = 1000
        self.scheduler = self._create_scheduler()
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=buffer_size)
        
        # 训练步数和性能追踪
        self.train_steps = 0
        self.episode_count = 0
        self.performance_history = deque(maxlen=100)  # 追踪最近100轮的表现
        
        # 噪声调度参数
        self.noise_schedule_enabled = True
        self.base_noise_scale = 1.0
        self.min_noise_scale = 0.1
        self.noise_decay_steps = 5000
        
        # 稳定化参数
        self.gradient_clip_norm = 1.0
        self.loss_smoothing_alpha = 0.1
        self.smoothed_loss = 0.0
        
        # 噪声重置频率控制
        self.noise_reset_frequency = 10  # 每10步重置一次噪声
        
    def _create_scheduler(self):
        """创建学习率调度器 - 预热 + 余弦退火"""
        def lr_lambda(step):
            if step < self.warmup_steps:
                # 预热阶段
                return step / self.warmup_steps
            else:
                # 余弦退火
                progress = (step - self.warmup_steps) / (10000 - self.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _update_noise_scale(self):
        """自适应噪声调度"""
        if not self.noise_schedule_enabled:
            return
        
        # 基于训练步数的噪声衰减
        progress = min(self.train_steps / self.noise_decay_steps, 1.0)
        noise_scale = self.min_noise_scale + (self.base_noise_scale - self.min_noise_scale) * (1 - progress)
        
        # 基于性能的自适应调整
        if len(self.performance_history) >= 20:
            recent_performance = np.mean(list(self.performance_history)[-20:])
            if recent_performance < 0.3:  # 表现较差时增加噪声
                noise_scale *= 1.2
            elif recent_performance > 0.8:  # 表现良好时减少噪声
                noise_scale *= 0.9
        
        # 设置噪声缩放
        self.q_network.set_noise_scale(noise_scale)
        self.target_network.set_noise_scale(noise_scale)
    
    def act(self, state, training=True):
        """选择动作"""
        state = torch.FloatTensor(state).to(device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state)
        self.q_network.train()
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done, info=None):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """训练智能体"""
        if len(self.memory) < self.batch_size:
            return
        
        # 采样经验
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值 - 使用Double DQN
        with torch.no_grad():
            # 使用主网络选择动作
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            # 使用目标网络评估Q值
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # 损失平滑
        self.smoothed_loss = self.loss_smoothing_alpha * loss.item() + (1 - self.loss_smoothing_alpha) * self.smoothed_loss
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip_norm)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # 更新噪声缩放
        self._update_noise_scale()
        
        # 控制噪声重置频率
        if self.train_steps % self.noise_reset_frequency == 0:
            self.q_network.reset_noise()
        
        # 更新目标网络
        if self.use_soft_update:
            self._soft_update_target_network()
        elif self.train_steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.train_steps += 1
    
    def _soft_update_target_network(self):
        """软更新目标网络 - 使用更大的tau值"""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def update_performance(self, success_rate):
        """更新性能历史"""
        self.performance_history.append(success_rate)
    
    def update_epsilon(self):
        """兼容性方法"""
        pass
    
    @property
    def epsilon(self):
        """兼容性属性"""
        return 0.0
    
    @property
    def current_noise_scale(self):
        """获取当前噪声缩放因子"""
        if hasattr(self.q_network.noisy1, 'noise_scale'):
            return self.q_network.noisy1.noise_scale
        return 1.0
    
    def get_training_stats(self):
        """获取训练统计信息"""
        return {
            'train_steps': self.train_steps,
            'current_lr': self.scheduler.get_last_lr()[0],
            'noise_scale': self.current_noise_scale,
            'smoothed_loss': self.smoothed_loss,
            'performance_trend': np.mean(list(self.performance_history)[-10:]) if len(self.performance_history) >= 10 else 0.0
        }
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_steps': self.train_steps,
            'performance_history': list(self.performance_history)
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'train_steps' in checkpoint:
            self.train_steps = checkpoint['train_steps']
        if 'performance_history' in checkpoint:
            self.performance_history = deque(checkpoint['performance_history'], maxlen=100) 