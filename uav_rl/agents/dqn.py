import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import math
import copy

# 设置种子以复现结果
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 检测设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 经验回放的数据结构
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# 标准经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity, batch_size=32, device='cpu'):
        self.capacity = capacity  # 容量上限
        self.buffer = deque(maxlen=capacity)  # 使用双端队列，自动管理容量
        self.batch_size = batch_size
        self.device = device
        self.position = 0  # 添加position属性以与PrioritizedReplayBuffer保持一致

    def add(self, state, action, reward, next_state, done):
        """添加经验到缓冲区
        
        Returns:
            int: 添加的经验在缓冲区中的位置
        """
        self.buffer.append((state, action, reward, next_state, done))
        current_position = self.position
        self.position = (self.position + 1) % self.capacity
        return current_position

    def sample(self, batch_size=None):
        """从经验回放缓冲区中随机采样
        
        Args:
            batch_size (int): 采样的批次大小
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones)的元组
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self) < batch_size:
            # 返回空数组而不是None，这样可以安全地进行拆包操作
            empty_array = np.array([])
            return (empty_array, empty_array, empty_array, empty_array, empty_array)
            
        indices = np.random.choice(len(self), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states)
        actions = np.array(actions).reshape(-1, 1)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)
        dones = np.array(dones).reshape(-1, 1)
        
        return states, actions, rewards, next_states, dones
    
    def sample_n(self, n):
        """从缓冲区采样指定数量的经验
        
        对于双缓冲区训练策略，需要从好经验和普通经验缓冲区分别采样不同数量的样本。
        
        Args:
            n (int): 需要采样的经验数量
            
        Returns:
            Tuple[ndarray]: 包含(states, actions, rewards, next_states, dones)的元组，全部转换为numpy数组
        """
        n = min(n, len(self.buffer))  # 确保n不超过缓冲区大小
        
        if n == 0:
            # 如果n为0，返回空数组
            return np.array([]), np.array([]).reshape(-1, 1), np.array([]).reshape(-1, 1), np.array([]), np.array([]).reshape(-1, 1)
            
        indices = np.random.choice(len(self.buffer), n, replace=False)
        
        # 解包采样的经验
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为numpy数组
        states = np.array(states)
        actions = np.array(actions).reshape(-1, 1)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)
        dones = np.array(dones).reshape(-1, 1)
        
        return states, actions, rewards, next_states, dones
        
    def truncate(self, new_size):
        """将缓冲区截断到指定大小，保留最新的样本
        
        Args:
            new_size (int): 新的缓冲区大小
        """
        if new_size < len(self.buffer):
            # 计算要保留的元素数量
            to_keep = new_size
            
            # 创建一个新的deque，并保留最新的to_keep个元素
            new_buffer = deque(list(self.buffer)[-to_keep:], maxlen=self.capacity)
            self.buffer = new_buffer
            
    def __len__(self):
        """获取当前缓冲区长度"""
        return len(self.buffer)

class DistributionalDQNetwork(nn.Module):
    def __init__(self, state_size, action_size, n_atoms=51, v_min=-10, v_max=10):
        super(DistributionalDQNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.support = torch.linspace(v_min, v_max, n_atoms)
        
        # 特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Dueling架构
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size * n_atoms)  # 每个动作有n_atoms个输出
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_atoms)  # 状态价值分布
        )
        
    def forward(self, x):
        """处理输入状态并生成Q值
        
        Args:
            x: 输入状态，可以是numpy数组或张量
            
        Returns:
            torch.Tensor: Q值张量
        """
        # 将输入转换为张量（如果不是）
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
            
        # 确保输入具有批次维度
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = self.feature_layer(x)
        
        # 计算价值和优势
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # 合并价值和优势: Q(s,a) = V(s) + (A(s,a) - 平均[A(s,a)])
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_q_values(self, x):
        """返回期望Q值（用于做决策）"""
        dist = self.forward(x)
        support = self.support.to(x.device)
        q_values = (dist * support).sum(dim=2)  # E[Z] = sum(p_i * z_i)
        return q_values

class AttentionMLP(nn.Module):
    """改进的MLP Self-Attention模块，更适合DQN环境"""
    def __init__(self, embed_dim, num_heads=1):
        super(AttentionMLP, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        # 简化的注意力机制，使用MLP而不是MultiheadAttention
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # 层归一化
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def forward(self, x):
        # x: (batch, embed_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # 自注意力机制
        q = self.query(x)  # (batch, embed_dim)
        k = self.key(x)    # (batch, embed_dim)
        v = self.value(x)  # (batch, embed_dim)
        
        # 计算注意力权重 (简化的点积注意力)
        attention_weights = torch.softmax(torch.sum(q * k, dim=1, keepdim=True), dim=1)
        attended = attention_weights * v
        
        # 残差连接和层归一化
        x = self.ln1(x + attended)
        
        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        return x

class DuelingDQNetwork(nn.Module):
    """改进的Dueling DQN网络结构，更适合UAV导航任务"""
    def __init__(self, state_size, action_size, hidden_size=256, use_attention=False, attn_heads=1):
        super(DuelingDQNetwork, self).__init__()
        self.use_attention = use_attention
        
        # 特征提取层 - 使用更深的网络
        self.features = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        if use_attention:
            self.attn = AttentionMLP(hidden_size // 2, attn_heads)
            
        # 价值流 - 更深的网络
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, 1)
        )
        
        # 优势流 - 更深的网络
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, action_size)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
                
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        features = self.features(state)
        
        if self.use_attention:
            features = self.attn(features)
            
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # 改进的Dueling组合方式
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class DQNNetwork(nn.Module):
    """改进的DQN网络架构，含批归一化和dropout"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        # 输入层和批归一化
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        
        # 隐藏层和批归一化
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        
        # 输出层
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # 初始化参数
        self._initialize_weights()
        
    def _initialize_weights(self):
        """使用He初始化优化网络权重"""
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.constant_(self.fc2.bias, 0.01)
        nn.init.constant_(self.fc3.bias, 0.01)
        
    def forward(self, state):
        # 将numpy数组转换为tensor
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(next(self.parameters()).device)
            
        # 确保输入具有批次维度
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # 检查是否只有单个样本且网络处于训练模式
        batch_size = state.size(0)
        if batch_size == 1 and self.training:
            # 对于单样本输入，临时切换到评估模式
            was_training = True
            self.eval()
        else:
            was_training = False
            
        x = F.relu(self.bn1(self.fc1(state)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        output = self.fc3(x)
        
        # 如果之前是训练模式，恢复
        if was_training:
            self.train()
            
        return output

class DeepQNetworkAgent:
    """基于深度 Q 网络的智能体。"""
    
    def __init__(self, state_size, action_size, hidden_size=128, buffer_size=10000, batch_size=32, gamma=0.99, 
                 lr=1e-4, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, update_target_every=10, tau=0.001, 
                 use_soft_update=True, use_double=False, grad_clip=1.0, device=None, use_cnn=False, local_map_size=5,
                 use_dueling=False, use_attention=False, attn_heads=1):
        """初始化网络、优化器与经验回放。"""
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_target_every = update_target_every
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.use_double = use_double
        self.grad_clip = grad_clip
        self.use_cnn = use_cnn
        self.local_map_size = local_map_size
        self.use_dueling = use_dueling
        self.use_attention = use_attention
        self.attn_heads = attn_heads
        
        # 确定设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # 创建Q网络和目标网络
        if self.use_dueling or self.use_attention:
            self.q_network = DuelingDQNetwork(state_size, action_size, hidden_size, use_attention=use_attention, attn_heads=attn_heads).to(self.device)
            self.target_network = DuelingDQNetwork(state_size, action_size, hidden_size, use_attention=use_attention, attn_heads=attn_heads).to(self.device)
        else:
            self.q_network = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
            self.target_network = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        
        # 复制参数到目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 设置优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # 创建经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size, batch_size, device=self.device)
        
        # 初始化训练步数
        self.train_steps = 0
        
    def update_learning_rate(self):
        """使用余弦退火策略更新学习率"""
        if not self.lr_scheduler_enabled:
            return
            
        # 计算当前周期内的位置 (0到1之间)
        position_in_cycle = (self.training_steps % self.lr_cycle_length) / self.lr_cycle_length
        
        # 使用余弦函数计算当前学习率因子 (在min_factor和max_factor之间)
        # 余弦从0到pi对应周期开始到结束
        cosine_factor = 0.5 * (1 + np.cos(position_in_cycle * np.pi))
        lr_factor = self.lr_min_factor + (self.lr_max_factor - self.lr_min_factor) * cosine_factor
        
        # 根据训练阶段，可能需要逐渐减小最大学习率
        training_progress = min(1.0, self.training_steps / self.max_training_steps) if hasattr(self, 'max_training_steps') else 0.5
        if training_progress > 0.7:  # 训练后期
            # 逐渐降低学习率上限
            max_factor_decay = max(0.5, 1.0 - (training_progress - 0.7) * 1.5)
            lr_factor *= max_factor_decay
        
        # 应用新的学习率
        new_lr = self.initial_lr * lr_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        # 检测周期结束，只在100轮时打印信息
        if self.training_steps > 0 and self.training_steps % self.lr_cycle_length == 0 and self.episodes_completed % 100 == 0:
            self.lr_cycle_count += 1
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.6f}")
            
            if self.lr_cycle_count % 3 == 0 and self.lr_cycle_length > 1000:
                self.lr_cycle_length = max(1000, int(self.lr_cycle_length * 0.8))
            
        return new_lr
    
    def save_best_model(self):
        """保存当前最佳模型状态"""
        self.best_model_state = {
            'q_network': copy.deepcopy(self.q_network.state_dict()),
            'target_network': copy.deepcopy(self.target_network.state_dict()),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episodes_completed': self.episodes_completed
        }
        
    def restore_best_model(self):
        """恢复到最佳模型状态"""
        if self.best_model_state is None:
            print("没有可恢复的最佳模型。")
            return False
            
        # 恢复网络参数
        self.q_network.load_state_dict(self.best_model_state['q_network'])
        self.target_network.load_state_dict(self.best_model_state['target_network'])
        
        # 只恢复epsilon，不恢复训练步数和episode计数
        # 这确保训练继续进行，但从更好的模型状态开始
        self.epsilon = self.best_model_state['epsilon']
        
        print(f"已恢复到最佳模型状态 (成功率: {self.best_success_rate:.2f})")
        return True
    
    def set_environment_info(self, start_pos, target_pos):
        """设置环境信息，用于目标导向探索"""
        self.start_pos = start_pos
        self.target_pos = target_pos
            
    def act(self, state, training=True):
        """选择动作，使用epsilon-greedy策略"""
        # 将状态转换为tensor
        state = torch.FloatTensor(state).to(self.device)
        
        # 在评估模式下不使用探索
        if not training:
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state)
            self.q_network.train()
            return q_values.argmax().item()
        
        # epsilon-greedy策略
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state)
            self.q_network.train()
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done, info=None):
        """记忆经验，使用增强的经验筛选和优先级机制"""
        # 将numpy数组转换为tensor
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)
        
        # 存储经验
        self.memory.add(state, action, reward, next_state, done)
        
        # 更新训练步数
        self.train_steps += 1
        
        # 每2000步检查一次并更新目标网络
        if self.train_steps % 2000 == 0:
            self.update_target_network()

    def train(self):
        """训练智能体"""
        # 如果经验不足，不进行训练
        if len(self.memory) < self.batch_size:
            return
            
        # 从经验回放中采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 将numpy数组转换为PyTorch张量
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            if self.use_double:
                # Double DQN
                next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # 普通DQN
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
            
        self.optimizer.step()
        
        # 软更新目标网络
        if self.use_soft_update:
            self._soft_update_target_network()
        
        # 更新训练步数
        self.train_steps += 1
        
        # 定期更新目标网络
        if not self.use_soft_update and self.train_steps % self.update_target_every == 0:
            self.update_target_network()

    def update_target_network(self):
        """硬更新目标网络 - 完全复制权重"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """加载模型"""
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

    def _soft_update_target_network(self):
        """软更新目标网络参数
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def update_epsilon(self):
        """更新探索率epsilon，使用简单的衰减策略"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def process_rewards(self, reward, done=False, episode_reward=None, info=None):
        """处理和调整奖励以提高学习稳定性
        
        改进版：动态奖励缩放、防止过拟合的奖励噪声以及训练阶段适应
        """
        # 如果没有传入episode_reward，使用类变量
        if episode_reward is None and hasattr(self, 'episode_reward'):
            episode_reward = self.episode_reward
            
        # 没有episode_reward时使用默认值
        if episode_reward is None:
            episode_reward = 0
            
        # 记录原始奖励用于调试
        original_reward = reward
        
        # 检查是否有额外信息
        if info is None:
            info = {}
            
        # 获取当前成功率
        success_rate = self.get_current_success_rate()
        
        # 获取训练进度
        progress = min(1.0, self.train_steps / self.max_train_steps) if hasattr(self, 'max_train_steps') and self.max_train_steps > 0 else 0.5
        
        # === 动态奖励缩放策略 ===
        
        # 基础奖励缩放 - 随训练进度变化
        if progress < 0.3:
            # 早期阶段 - 较大奖励刺激
            reward_scale = 1.8
        elif progress < 0.7:
            # 中期阶段 - 适中奖励
            reward_scale = 1.5
        else:
            # 后期阶段 - 较小奖励以防过拟合
            reward_scale = 1.2
            
        # 应用基础缩放
        reward = reward * reward_scale
        
        # === 防止过拟合的随机奖励噪声 ===
        
        # 在非终止状态下添加小噪声，防止过拟合特定轨迹
        if not done and progress > 0.4:  # 中后期训练才添加噪声
            noise_magnitude = 0.05 + progress * 0.1  # 随训练进度增加噪声
            noise = np.random.normal(0, noise_magnitude)
            reward = reward + noise
            
        # === 根据环境信息调整奖励 ===
            
        # 根据距离信息调整奖励（如果有）
        if 'distance_to_target' in info:
            distance = info['distance_to_target']
            # 动态距离因子，随训练进展，对距离的敏感度降低
            sensitivity = 1.0 - progress * 0.5  # 从1.0降低到0.5
            distance_factor = min(1.0, max(0.2, 1.0 - (distance / 10.0) * sensitivity))
            reward = reward * distance_factor
            
        # 对重复动作的惩罚 - 随训练进展增加惩罚
        if 'repeated_action_count' in info and info['repeated_action_count'] > 2:
            repeat_count = info['repeated_action_count']
            base_penalty = 0.1 + progress * 0.1  # 随训练进展增加基础惩罚
            repeat_penalty = min(0.7, repeat_count * base_penalty)  # 增加最大惩罚
            reward = reward * (1.0 - repeat_penalty)
            
        # === 成功率特殊处理 ===
            
        # 成功率调整 - 在不同阶段采用不同策略
        if success_rate < 0.2:
            # 低成功率 - 加强正向奖励，减轻负向惩罚
            if reward > 0:
                reward = reward * 1.3  # 强化正向奖励
            elif reward < 0:
                reward = reward * 0.8  # 减轻负向惩罚
        elif success_rate > 0.8 and progress > 0.5:
            # 高成功率且训练后期 - 增加挑战性
            if reward > 0:
                reward = reward * 0.9  # 降低正向奖励
            elif reward < 0:
                reward = reward * 1.2  # 加强负向惩罚
                
        # === 终止状态特殊处理 ===
                
        # 处理终止状态的特殊奖励
        if done:
            # 成功终止获得额外奖励
            if 'success' in info and info['success']:
                # 成功奖励随着成功率提高而递减，防止过拟合
                base_success_bonus = 2.0
                success_bonus = base_success_bonus * (1.0 - success_rate * 0.7)
                reward += success_bonus
            # 失败终止的额外惩罚（可选）
            elif success_rate > 0.5:  # 只在成功率高时对失败增加额外惩罚
                failure_penalty = -1.0 * success_rate  # 成功率越高，失败惩罚越大
                reward += failure_penalty
        
        # === 奖励裁剪与归一化 ===
            
        # 动态奖励上下限，随训练进度调整
        reward_max = 8.0 - progress * 3.0  # 从8.0降至5.0
        reward_min = -6.0 + progress * 2.0  # 从-6.0升至-4.0
        
        # 奖励裁剪 - 使用动态范围
        reward = max(reward_min, min(reward_max, reward))
            
        # 只在调试模式下打印奖励调整详情
        if hasattr(self, 'debug') and self.debug and abs(reward - original_reward) > 0.2:
            print(f"奖励调整: {original_reward:.2f} → {reward:.2f}, 进度: {progress:.2f}, 成功率: {success_rate:.2f}")
            
        return reward

    def choose_action(self, state, eval_mode=False, current_position=None, target_position=None):
        """选择动作，增强型智能选择与多样化探索
        
        Args:
            state: 当前状态
            eval_mode: 是否处于评估模式
            current_position: 当前位置坐标 (x, y)
            target_position: 目标位置坐标 (x, y)
            
        Returns:
            int: 选择的动作索引
        """
        # 保存操作模式，便于评估后恢复
        was_training = self.q_network.training
        
        # 设置为评估模式进行预测
        self.q_network.eval()
        
        # 初始化训练进度 (如果不存在)
        if not hasattr(self, 'training_progress'):
            self.training_progress = 0.0
        else:
            self.training_progress = min(1.0, self.train_steps / self.max_train_steps) if hasattr(self, 'max_train_steps') and self.max_train_steps > 0 else 0.5
            
        # 获取当前成功率 (如果不存在)
        success_rate = 0.0
        if hasattr(self, 'get_current_success_rate'):
            success_rate = self.get_current_success_rate()
        
        # 确定当前使用的探索率
        if eval_mode:
            # 评估模式下使用极低的探索率
            current_epsilon = 0.02
        else:
            # 动态调整epsilon的实际应用值
            # 1. 根据训练进度降低基础值
            base_epsilon = self.epsilon
            
            # 2. 根据成功率动态调整
            if success_rate > 0.7 and self.training_progress > 0.5:
                # 高成功率且训练后期，略微降低以减少探索
                current_epsilon = base_epsilon * 0.85
            elif success_rate < 0.2:
                # 低成功率，稍微提高以增加探索
                current_epsilon = min(0.9, base_epsilon * 1.15)
            else:
                current_epsilon = base_epsilon
        
        # 初始化动作历史记录（如果不存在）
        if not hasattr(self, 'last_actions'):
            self.last_actions = []
            
        # 初始化动作重复计数器（如果不存在）
        if not hasattr(self, 'action_repeat_counts'):
            self.action_repeat_counts = {}
            
        # 更新动作重复计数器
        action_key = str(current_position)
        if action_key not in self.action_repeat_counts:
            self.action_repeat_counts[action_key] = [0] * self.action_size
            
        # 获取状态的Q值
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
            
        # 最近使用过的动作
        recent_actions = self.last_actions[-5:] if len(self.last_actions) > 0 else []
        
        # 计算实际的探索策略比例
        random_explore_ratio = 0.2  # 纯随机探索比例
        smart_explore_ratio = 0.6   # 智能探索比例
        q_value_ratio = 0.2        # Q值选择比例
        
        # 训练后期调整探索策略比例
        if self.training_progress > 0.7:
            random_explore_ratio = 0.1
            smart_explore_ratio = 0.4
            q_value_ratio = 0.5
            
        # 探索阶段的决策
        if np.random.random() < current_epsilon:
            chosen_action = None
            
            # 决定使用哪种探索策略
            strategy_rand = np.random.random()
            
            # 1. 纯随机探索
            if strategy_rand < random_explore_ratio:
                chosen_action = np.random.randint(0, self.action_size)
                
            # 2. 智能探索 - 目标导向或避免重复
            elif strategy_rand < random_explore_ratio + smart_explore_ratio:
                # 如果有位置信息，尝试目标导向探索
                if current_position is not None and target_position is not None:
                    # 80%概率使用目标导向
                    if np.random.random() < 0.8:
                        chosen_action = self._choose_direction_action(current_position, target_position)
                    else:
                        # 避免重复动作
                        repeat_counts = self.action_repeat_counts[action_key]
                        chosen_action = self._choose_non_repeating_action(repeat_counts)
                else:
                    # 没有位置信息时，避免重复动作
                    repeat_counts = self.action_repeat_counts[action_key] if action_key in self.action_repeat_counts else [0] * self.action_size
                    chosen_action = self._choose_non_repeating_action(repeat_counts)
            
            # 3. 基于Q值的带噪声选择 - 有策略地探索
            else:
                # 添加高斯噪声到Q值
                noise_scale = 0.2 + (1.0 - self.training_progress) * 0.3  # 随训练进度降低噪声
                noisy_q = q_values + np.random.normal(0, noise_scale, size=q_values.shape)
                
                # 避免多次重复相同动作
                for i in range(len(noisy_q)):
                    if i in recent_actions:
                        # 根据出现次数降低Q值
                        count = recent_actions.count(i)
                        noisy_q[i] -= count * 0.3  # 每次重复降低0.3
                
                chosen_action = np.argmax(noisy_q)
        else:
            # 利用阶段 - 使用Q值选择最佳动作
            
            # 在高成功率且训练晚期，添加小量随机性防止过拟合
            if success_rate > 0.8 and self.training_progress > 0.7 and not eval_mode:
                # 添加小量噪声到Q值
                small_noise = np.random.normal(0, 0.05, size=q_values.shape)
                q_values = q_values + small_noise
                
            # 避免连续重复相同动作多次 (训练中后期才启用)
            if self.training_progress > 0.4 and not eval_mode:
                repeat_penalty = 0.1 * self.training_progress  # 随训练进展增加惩罚
                for i in range(len(q_values)):
                    # 检查最近3个动作
                    if len(self.last_actions) >= 3 and all(a == i for a in self.last_actions[-3:]):
                        q_values[i] -= repeat_penalty * 3  # 惩罚连续重复3次的动作
                    # 检查最近的动作
                    elif len(self.last_actions) > 0 and self.last_actions[-1] == i:
                        q_values[i] -= repeat_penalty  # 轻微惩罚刚刚使用的动作
                
            chosen_action = np.argmax(q_values)
        
        # 记录所选动作并更新重复计数
        if not eval_mode:
            self.last_actions.append(chosen_action)
            # 限制历史长度
            if len(self.last_actions) > 20:
                self.last_actions.pop(0)
                
            # 更新重复计数
            if action_key in self.action_repeat_counts:
                self.action_repeat_counts[action_key][chosen_action] += 1
                
                # 定期重置计数以避免无限增长
                if np.sum(self.action_repeat_counts[action_key]) > 100:
                    self.action_repeat_counts[action_key] = [max(0, count//2) for count in self.action_repeat_counts[action_key]]
        
        # 恢复网络模式
        if was_training:
            self.q_network.train()
            
        return chosen_action
        
    def _choose_direction_action(self, current_position, target_position):
        """基于当前位置和目标位置选择方向性动作
        
        Args:
            current_position: 当前位置坐标 (x, y)
            target_position: 目标位置坐标 (x, y)
            
        Returns:
            int: 选择的动作索引
        """
        x_diff = target_position[0] - current_position[0]
        y_diff = target_position[1] - current_position[1]
        
        # 基础动作概率
        probs = np.ones(self.action_size) * 0.05
        
        # 假设动作映射: 0=上, 1=右, 2=下, 3=左, (4=停留-如果有的话)
        
        # 确定主要移动方向
        if abs(x_diff) > abs(y_diff):
            # 主要在水平方向移动
            if x_diff > 0:  # 目标在右边
                probs[1] += 0.7  # 提高向右移动概率
            else:  # 目标在左边
                probs[3] += 0.7  # 提高向左移动概率
                
            # 次要垂直方向
            if y_diff > 0:  # 目标偏下
                probs[2] += 0.1  # 向下移动有一定概率
            elif y_diff < 0:  # 目标偏上
                probs[0] += 0.1  # 向上移动有一定概率
        else:
            # 主要在垂直方向移动
            if y_diff > 0:  # 目标在下方
                probs[2] += 0.7  # 提高向下移动概率
            else:  # 目标在上方
                probs[0] += 0.7  # 提高向上移动概率
                
            # 次要水平方向
            if x_diff > 0:  # 目标偏右
                probs[1] += 0.1  # 向右移动有一定概率
            elif x_diff < 0:  # 目标偏左
                probs[3] += 0.1  # 向左移动有一定概率
                
        # 减少选择重复动作的概率
        if len(self.last_actions) >= 2:
            last_action = self.last_actions[-1]
            prev_action = self.last_actions[-2]
            
            # 如果连续相同动作，大幅降低其概率
            if last_action == prev_action:
                probs[last_action] *= 0.3
                
        # 归一化概率
        probs = probs / np.sum(probs)
        
        # 按概率选择动作
        return np.random.choice(self.action_size, p=probs)
        
    def _choose_non_repeating_action(self, repeat_counts):
        """选择非重复动作，避免动作循环
        
        Args:
            repeat_counts: 各动作的重复计数
            
        Returns:
            int: 选择的动作索引
        """
        # 反转计数为选择概率（计数越低，概率越高）
        if np.sum(repeat_counts) > 0:
            probs = 1.0 / (np.array(repeat_counts) + 1.0)
            probs = probs / np.sum(probs)  # 归一化
            return np.random.choice(self.action_size, p=probs)
        else:
            return np.random.randint(0, self.action_size)
        
    def _combine_experiences(self, exp1, exp2):
        """合并两组经验
        
        Args:
            exp1: 第一组经验元组
            exp2: 第二组经验元组
            
        Returns:
            tuple: 合并后的经验元组
        """
        # 解包两组经验
        states1, actions1, rewards1, next_states1, dones1 = exp1
        states2, actions2, rewards2, next_states2, dones2 = exp2
        
        # 合并经验
        states = np.concatenate([states1, states2])
        actions = np.concatenate([actions1, actions2])
        rewards = np.concatenate([rewards1, rewards2])
        next_states = np.concatenate([next_states1, next_states2])
        dones = np.concatenate([dones1, dones2])
        
        return states, actions, rewards, next_states, dones