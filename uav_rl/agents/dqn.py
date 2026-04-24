import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import math
import copy

# English comment.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# English comment.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# English comment.
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# English comment.
class ReplayBuffer:
    def __init__(self, capacity, batch_size=32, device='cpu'):
        self.capacity = capacity  # English comment.
        self.buffer = deque(maxlen=capacity)  # English comment.
        self.batch_size = batch_size
        self.device = device
        self.position = 0  # English comment.

    def add(self, state, action, reward, next_state, done):
        """English documentation."""
        
        Returns:
            English documentation.
        """
        self.buffer.append((state, action, reward, next_state, done))
        current_position = self.position
        self.position = (self.position + 1) % self.capacity
        return current_position

    def sample(self, batch_size=None):
        """English documentation."""
        
        Args:
            English documentation.
            
        Returns:
            English documentation.
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self) < batch_size:
            # English comment.
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
        """English documentation."""
        
        English documentation.
        
        Args:
            English documentation.
            
        Returns:
            English documentation.
        """
        n = min(n, len(self.buffer))  # English comment.
        
        if n == 0:
            # English comment.
            return np.array([]), np.array([]).reshape(-1, 1), np.array([]).reshape(-1, 1), np.array([]), np.array([]).reshape(-1, 1)
            
        indices = np.random.choice(len(self.buffer), n, replace=False)
        
        # English comment.
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # English comment.
        states = np.array(states)
        actions = np.array(actions).reshape(-1, 1)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)
        dones = np.array(dones).reshape(-1, 1)
        
        return states, actions, rewards, next_states, dones
        
    def truncate(self, new_size):
        """English documentation."""
        
        Args:
            English documentation.
        """
        if new_size < len(self.buffer):
            # English comment.
            to_keep = new_size
            
            # English comment.
            new_buffer = deque(list(self.buffer)[-to_keep:], maxlen=self.capacity)
            self.buffer = new_buffer
            
    def __len__(self):
        """English documentation."""
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
        
        # English comment.
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # English comment.
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size * n_atoms)  # English comment.
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_atoms)  # English comment.
        )
        
    def forward(self, x):
        """English documentation."""
        
        Args:
            English documentation.
            
        Returns:
            English documentation.
        """
        # English comment.
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
            
        # English comment.
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = self.feature_layer(x)
        
        # English comment.
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # English comment.
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_q_values(self, x):
        """English documentation."""
        dist = self.forward(x)
        support = self.support.to(x.device)
        q_values = (dist * support).sum(dim=2)  # E[Z] = sum(p_i * z_i)
        return q_values

class AttentionMLP(nn.Module):
    """English documentation."""
    def __init__(self, embed_dim, num_heads=1):
        super(AttentionMLP, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        # English comment.
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # English comment.
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # English comment.
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # English comment.
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
            
        # English comment.
        q = self.query(x)  # (batch, embed_dim)
        k = self.key(x)    # (batch, embed_dim)
        v = self.value(x)  # (batch, embed_dim)
        
        # English comment.
        attention_weights = torch.softmax(torch.sum(q * k, dim=1, keepdim=True), dim=1)
        attended = attention_weights * v
        
        # English comment.
        x = self.ln1(x + attended)
        
        # English comment.
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        return x

class DuelingDQNetwork(nn.Module):
    """English documentation."""
    def __init__(self, state_size, action_size, hidden_size=256, use_attention=False, attn_heads=1):
        super(DuelingDQNetwork, self).__init__()
        self.use_attention = use_attention
        
        # English comment.
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
            
        # English comment.
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, 1)
        )
        
        # English comment.
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
        
        # English comment.
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class DQNNetwork(nn.Module):
    """English documentation."""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        # English comment.
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        
        # English comment.
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        
        # English comment.
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # English comment.
        self._initialize_weights()
        
    def _initialize_weights(self):
        """English documentation."""
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.constant_(self.fc2.bias, 0.01)
        nn.init.constant_(self.fc3.bias, 0.01)
        
    def forward(self, state):
        # English comment.
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(next(self.parameters()).device)
            
        # English comment.
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # English comment.
        batch_size = state.size(0)
        if batch_size == 1 and self.training:
            # English comment.
            was_training = True
            self.eval()
        else:
            was_training = False
            
        x = F.relu(self.bn1(self.fc1(state)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        output = self.fc3(x)
        
        # English comment.
        if was_training:
            self.train()
            
        return output

class DeepQNetworkAgent:
    """English documentation."""
    
    def __init__(self, state_size, action_size, hidden_size=128, buffer_size=10000, batch_size=32, gamma=0.99, 
                 lr=1e-4, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, update_target_every=10, tau=0.001, 
                 use_soft_update=True, use_double=False, grad_clip=1.0, device=None, use_cnn=False, local_map_size=5,
                 use_dueling=False, use_attention=False, attn_heads=1):
        """English documentation."""
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
        
        # English comment.
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # English comment.
        if self.use_dueling or self.use_attention:
            self.q_network = DuelingDQNetwork(state_size, action_size, hidden_size, use_attention=use_attention, attn_heads=attn_heads).to(self.device)
            self.target_network = DuelingDQNetwork(state_size, action_size, hidden_size, use_attention=use_attention, attn_heads=attn_heads).to(self.device)
        else:
            self.q_network = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
            self.target_network = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        
        # English comment.
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # English comment.
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # English comment.
        self.memory = ReplayBuffer(buffer_size, batch_size, device=self.device)
        
        # English comment.
        self.train_steps = 0
        
    def update_learning_rate(self):
        """English documentation."""
        if not self.lr_scheduler_enabled:
            return
            
        # English comment.
        position_in_cycle = (self.training_steps % self.lr_cycle_length) / self.lr_cycle_length
        
        # English comment.
        # English comment.
        cosine_factor = 0.5 * (1 + np.cos(position_in_cycle * np.pi))
        lr_factor = self.lr_min_factor + (self.lr_max_factor - self.lr_min_factor) * cosine_factor
        
        # English comment.
        training_progress = min(1.0, self.training_steps / self.max_training_steps) if hasattr(self, 'max_training_steps') else 0.5
        if training_progress > 0.7:  # English comment.
            # English comment.
            max_factor_decay = max(0.5, 1.0 - (training_progress - 0.7) * 1.5)
            lr_factor *= max_factor_decay
        
        # English comment.
        new_lr = self.initial_lr * lr_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        # English comment.
        if self.training_steps > 0 and self.training_steps % self.lr_cycle_length == 0 and self.episodes_completed % 100 == 0:
            self.lr_cycle_count += 1
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f": {current_lr:.6f}")
            
            if self.lr_cycle_count % 3 == 0 and self.lr_cycle_length > 1000:
                self.lr_cycle_length = max(1000, int(self.lr_cycle_length * 0.8))
            
        return new_lr
    
    def save_best_model(self):
        """English documentation."""
        self.best_model_state = {
            'q_network': copy.deepcopy(self.q_network.state_dict()),
            'target_network': copy.deepcopy(self.target_network.state_dict()),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episodes_completed': self.episodes_completed
        }
        
    def restore_best_model(self):
        """English documentation."""
        if self.best_model_state is None:
            print("。")
            return False
            
        # English comment.
        self.q_network.load_state_dict(self.best_model_state['q_network'])
        self.target_network.load_state_dict(self.best_model_state['target_network'])
        
        # English comment.
        # English comment.
        self.epsilon = self.best_model_state['epsilon']
        
        print(f" (: {self.best_success_rate:.2f})")
        return True
    
    def set_environment_info(self, start_pos, target_pos):
        """English documentation."""
        self.start_pos = start_pos
        self.target_pos = target_pos
            
    def act(self, state, training=True):
        """English documentation."""
        # English comment.
        state = torch.FloatTensor(state).to(self.device)
        
        # English comment.
        if not training:
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state)
            self.q_network.train()
            return q_values.argmax().item()
        
        # English comment.
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state)
            self.q_network.train()
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done, info=None):
        """English documentation."""
        # English comment.
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)
        
        # English comment.
        self.memory.add(state, action, reward, next_state, done)
        
        # English comment.
        self.train_steps += 1
        
        # English comment.
        if self.train_steps % 2000 == 0:
            self.update_target_network()

    def train(self):
        """English documentation."""
        # English comment.
        if len(self.memory) < self.batch_size:
            return
            
        # English comment.
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # English comment.
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # English comment.
        current_q_values = self.q_network(states).gather(1, actions)
        
        # English comment.
        with torch.no_grad():
            if self.use_double:
                # Double DQN
                next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # English comment.
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # English comment.
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # English comment.
        self.optimizer.zero_grad()
        loss.backward()
        
        # English comment.
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
            
        self.optimizer.step()
        
        # English comment.
        if self.use_soft_update:
            self._soft_update_target_network()
        
        # English comment.
        self.train_steps += 1
        
        # English comment.
        if not self.use_soft_update and self.train_steps % self.update_target_every == 0:
            self.update_target_network()

    def update_target_network(self):
        """English documentation."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path):
        """English documentation."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """English documentation."""
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

    def _soft_update_target_network(self):
        """English documentation."""
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def update_epsilon(self):
        """English documentation."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def process_rewards(self, reward, done=False, episode_reward=None, info=None):
        """English documentation."""
        
        English documentation.
        """
        # English comment.
        if episode_reward is None and hasattr(self, 'episode_reward'):
            episode_reward = self.episode_reward
            
        # English comment.
        if episode_reward is None:
            episode_reward = 0
            
        # English comment.
        original_reward = reward
        
        # English comment.
        if info is None:
            info = {}
            
        # English comment.
        success_rate = self.get_current_success_rate()
        
        # English comment.
        progress = min(1.0, self.train_steps / self.max_train_steps) if hasattr(self, 'max_train_steps') and self.max_train_steps > 0 else 0.5
        
        # English comment.
        
        # English comment.
        if progress < 0.3:
            # English comment.
            reward_scale = 1.8
        elif progress < 0.7:
            # English comment.
            reward_scale = 1.5
        else:
            # English comment.
            reward_scale = 1.2
            
        # English comment.
        reward = reward * reward_scale
        
        # English comment.
        
        # English comment.
        if not done and progress > 0.4:  # English comment.
            noise_magnitude = 0.05 + progress * 0.1  # English comment.
            noise = np.random.normal(0, noise_magnitude)
            reward = reward + noise
            
        # English comment.
            
        # English comment.
        if 'distance_to_target' in info:
            distance = info['distance_to_target']
            # English comment.
            sensitivity = 1.0 - progress * 0.5  # English comment.
            distance_factor = min(1.0, max(0.2, 1.0 - (distance / 10.0) * sensitivity))
            reward = reward * distance_factor
            
        # English comment.
        if 'repeated_action_count' in info and info['repeated_action_count'] > 2:
            repeat_count = info['repeated_action_count']
            base_penalty = 0.1 + progress * 0.1  # English comment.
            repeat_penalty = min(0.7, repeat_count * base_penalty)  # English comment.
            reward = reward * (1.0 - repeat_penalty)
            
        # English comment.
            
        # English comment.
        if success_rate < 0.2:
            # English comment.
            if reward > 0:
                reward = reward * 1.3  # English comment.
            elif reward < 0:
                reward = reward * 0.8  # English comment.
        elif success_rate > 0.8 and progress > 0.5:
            # English comment.
            if reward > 0:
                reward = reward * 0.9  # English comment.
            elif reward < 0:
                reward = reward * 1.2  # English comment.
                
        # English comment.
                
        # English comment.
        if done:
            # English comment.
            if 'success' in info and info['success']:
                # English comment.
                base_success_bonus = 2.0
                success_bonus = base_success_bonus * (1.0 - success_rate * 0.7)
                reward += success_bonus
            # English comment.
            elif success_rate > 0.5:  # English comment.
                failure_penalty = -1.0 * success_rate  # English comment.
                reward += failure_penalty
        
        # English comment.
            
        # English comment.
        reward_max = 8.0 - progress * 3.0  # English comment.
        reward_min = -6.0 + progress * 2.0  # English comment.
        
        # English comment.
        reward = max(reward_min, min(reward_max, reward))
            
        # English comment.
        if hasattr(self, 'debug') and self.debug and abs(reward - original_reward) > 0.2:
            print(f": {original_reward:.2f} → {reward:.2f}, : {progress:.2f}, : {success_rate:.2f}")
            
        return reward

    def choose_action(self, state, eval_mode=False, current_position=None, target_position=None):
        """English documentation."""
        
        Args:
            English documentation.
            English documentation.
            English documentation.
            English documentation.
            
        Returns:
            English documentation.
        """
        # English comment.
        was_training = self.q_network.training
        
        # English comment.
        self.q_network.eval()
        
        # English comment.
        if not hasattr(self, 'training_progress'):
            self.training_progress = 0.0
        else:
            self.training_progress = min(1.0, self.train_steps / self.max_train_steps) if hasattr(self, 'max_train_steps') and self.max_train_steps > 0 else 0.5
            
        # English comment.
        success_rate = 0.0
        if hasattr(self, 'get_current_success_rate'):
            success_rate = self.get_current_success_rate()
        
        # English comment.
        if eval_mode:
            # English comment.
            current_epsilon = 0.02
        else:
            # English comment.
            # English comment.
            base_epsilon = self.epsilon
            
            # English comment.
            if success_rate > 0.7 and self.training_progress > 0.5:
                # English comment.
                current_epsilon = base_epsilon * 0.85
            elif success_rate < 0.2:
                # English comment.
                current_epsilon = min(0.9, base_epsilon * 1.15)
            else:
                current_epsilon = base_epsilon
        
        # English comment.
        if not hasattr(self, 'last_actions'):
            self.last_actions = []
            
        # English comment.
        if not hasattr(self, 'action_repeat_counts'):
            self.action_repeat_counts = {}
            
        # English comment.
        action_key = str(current_position)
        if action_key not in self.action_repeat_counts:
            self.action_repeat_counts[action_key] = [0] * self.action_size
            
        # English comment.
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
            
        # English comment.
        recent_actions = self.last_actions[-5:] if len(self.last_actions) > 0 else []
        
        # English comment.
        random_explore_ratio = 0.2  # English comment.
        smart_explore_ratio = 0.6  # English comment.
        q_value_ratio = 0.2  # English comment.
        
        # English comment.
        if self.training_progress > 0.7:
            random_explore_ratio = 0.1
            smart_explore_ratio = 0.4
            q_value_ratio = 0.5
            
        # English comment.
        if np.random.random() < current_epsilon:
            chosen_action = None
            
            # English comment.
            strategy_rand = np.random.random()
            
            # English comment.
            if strategy_rand < random_explore_ratio:
                chosen_action = np.random.randint(0, self.action_size)
                
            # English comment.
            elif strategy_rand < random_explore_ratio + smart_explore_ratio:
                # English comment.
                if current_position is not None and target_position is not None:
                    # English comment.
                    if np.random.random() < 0.8:
                        chosen_action = self._choose_direction_action(current_position, target_position)
                    else:
                        # English comment.
                        repeat_counts = self.action_repeat_counts[action_key]
                        chosen_action = self._choose_non_repeating_action(repeat_counts)
                else:
                    # English comment.
                    repeat_counts = self.action_repeat_counts[action_key] if action_key in self.action_repeat_counts else [0] * self.action_size
                    chosen_action = self._choose_non_repeating_action(repeat_counts)
            
            # English comment.
            else:
                # English comment.
                noise_scale = 0.2 + (1.0 - self.training_progress) * 0.3  # English comment.
                noisy_q = q_values + np.random.normal(0, noise_scale, size=q_values.shape)
                
                # English comment.
                for i in range(len(noisy_q)):
                    if i in recent_actions:
                        # English comment.
                        count = recent_actions.count(i)
                        noisy_q[i] -= count * 0.3  # English comment.
                
                chosen_action = np.argmax(noisy_q)
        else:
            # English comment.
            
            # English comment.
            if success_rate > 0.8 and self.training_progress > 0.7 and not eval_mode:
                # English comment.
                small_noise = np.random.normal(0, 0.05, size=q_values.shape)
                q_values = q_values + small_noise
                
            # English comment.
            if self.training_progress > 0.4 and not eval_mode:
                repeat_penalty = 0.1 * self.training_progress  # English comment.
                for i in range(len(q_values)):
                    # English comment.
                    if len(self.last_actions) >= 3 and all(a == i for a in self.last_actions[-3:]):
                        q_values[i] -= repeat_penalty * 3  # English comment.
                    # English comment.
                    elif len(self.last_actions) > 0 and self.last_actions[-1] == i:
                        q_values[i] -= repeat_penalty  # English comment.
                
            chosen_action = np.argmax(q_values)
        
        # English comment.
        if not eval_mode:
            self.last_actions.append(chosen_action)
            # English comment.
            if len(self.last_actions) > 20:
                self.last_actions.pop(0)
                
            # English comment.
            if action_key in self.action_repeat_counts:
                self.action_repeat_counts[action_key][chosen_action] += 1
                
                # English comment.
                if np.sum(self.action_repeat_counts[action_key]) > 100:
                    self.action_repeat_counts[action_key] = [max(0, count//2) for count in self.action_repeat_counts[action_key]]
        
        # English comment.
        if was_training:
            self.q_network.train()
            
        return chosen_action
        
    def _choose_direction_action(self, current_position, target_position):
        """English documentation."""
        
        Args:
            English documentation.
            English documentation.
            
        Returns:
            English documentation.
        """
        x_diff = target_position[0] - current_position[0]
        y_diff = target_position[1] - current_position[1]
        
        # English comment.
        probs = np.ones(self.action_size) * 0.05
        
        # English comment.
        
        # English comment.
        if abs(x_diff) > abs(y_diff):
            # English comment.
            if x_diff > 0:  # English comment.
                probs[1] += 0.7  # English comment.
            else:  # English comment.
                probs[3] += 0.7  # English comment.
                
            # English comment.
            if y_diff > 0:  # English comment.
                probs[2] += 0.1  # English comment.
            elif y_diff < 0:  # English comment.
                probs[0] += 0.1  # English comment.
        else:
            # English comment.
            if y_diff > 0:  # English comment.
                probs[2] += 0.7  # English comment.
            else:  # English comment.
                probs[0] += 0.7  # English comment.
                
            # English comment.
            if x_diff > 0:  # English comment.
                probs[1] += 0.1  # English comment.
            elif x_diff < 0:  # English comment.
                probs[3] += 0.1  # English comment.
                
        # English comment.
        if len(self.last_actions) >= 2:
            last_action = self.last_actions[-1]
            prev_action = self.last_actions[-2]
            
            # English comment.
            if last_action == prev_action:
                probs[last_action] *= 0.3
                
        # English comment.
        probs = probs / np.sum(probs)
        
        # English comment.
        return np.random.choice(self.action_size, p=probs)
        
    def _choose_non_repeating_action(self, repeat_counts):
        """English documentation."""
        
        Args:
            English documentation.
            
        Returns:
            English documentation.
        """
        # English comment.
        if np.sum(repeat_counts) > 0:
            probs = 1.0 / (np.array(repeat_counts) + 1.0)
            probs = probs / np.sum(probs)  # English comment.
            return np.random.choice(self.action_size, p=probs)
        else:
            return np.random.randint(0, self.action_size)
        
    def _combine_experiences(self, exp1, exp2):
        """English documentation."""
        
        Args:
            English documentation.
            English documentation.
            
        Returns:
            English documentation.
        """
        # English comment.
        states1, actions1, rewards1, next_states1, dones1 = exp1
        states2, actions2, rewards2, next_states2, dones2 = exp2
        
        # English comment.
        states = np.concatenate([states1, states2])
        actions = np.concatenate([actions1, actions2])
        rewards = np.concatenate([rewards1, rewards2])
        next_states = np.concatenate([next_states1, next_states2])
        dones = np.concatenate([dones1, dones2])
        
        return states, actions, rewards, next_states, dones