import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gym import Env
from gym.spaces import Discrete, Box
import random

class UAVGridWorldEnvironment(Env):
    """栅格世界中的 UAV 导航环境（障碍物、弱信号区与局部观测）。"""

    def __init__(self, grid_size=15, max_steps=40, difficulty='medium', use_local_map=True, local_map_size=5):
        super(UAVGridWorldEnvironment, self).__init__()
        
        # Grid world parameters
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.difficulty = difficulty
        
        self.use_local_map = use_local_map
        self.local_map_size = local_map_size
        self.debug = False
        
        # UAV state
        self.uav_pos = [0, 0]  # Start position (default)
        self.prev_pos = [0, 0]
        self.start_pos = [0, 0]
        
        self.target_pos = [14, 14]
        
        # 生成障碍物和低信号区域
        self.obstacles = self._generate_obstacles()
        self.low_signal_areas = self._calculate_low_signal_areas()
        
        self.fog_of_war = False
        self.visible_range = grid_size
        self.steps_taken = 0
        
        # 动作空间: 0=上, 1=右, 2=下, 3=左, 4=停留
        self.action_space = Discrete(5)
        
        if self.use_local_map:
            local_map_features = self.local_map_size * self.local_map_size
            global_features = 6
            self.observation_space = Box(
                low=np.array([0.0] * local_map_features + [-grid_size, -grid_size, 0.0, 0.0, 0.0, -1000.0], dtype=np.float32),
                high=np.array([1.0] * local_map_features + [grid_size, grid_size, grid_size*np.sqrt(2), 1.0, 1.0, 1000.0], dtype=np.float32),
                shape=(local_map_features + global_features,),
                dtype=np.float32
            )
        else:
            self.observation_space = Box(
                low=np.array([-grid_size, -grid_size, 0, 0.0, 0.0, -1000.0], dtype=np.float32),
                high=np.array([grid_size, grid_size, grid_size*np.sqrt(2), 1.0, 1.0, 1000.0], dtype=np.float32),
                shape=(6,),
                dtype=np.float32
            )
        
        self.reached_target = False
        self.prev_distance = 0
        self.initial_distance = 0
        self.prev_reward = 0.0
        self.episode_count = 0
        self.success_history = []
        self.success_window_size = 10
        
        self.trajectory = []
        self.visited_positions = set()
        self.state = None
        self.prev_signal_strength = None
        self.reset()
    
    def set_training_params(self, total_episodes=200):
        self.total_episodes = total_episodes

    def _generate_obstacles(self):
        """生成多通道障碍物布局。"""
        obstacles = []
        for x in [2, 3, 5]:
            for y in range(2, 13):
                if y not in [7, 8]:
                    obstacles.append([x, y])
        for x in [7, 9]:
            for y in range(2, 13):
                if y not in [5, 6, 7, 8, 9]:
                    obstacles.append([x, y])
        for x in [10, 11, 13]:
            for y in range(2, 13):
                if y not in [7, 8]:
                    obstacles.append([x, y])
        horizontal_obstacles = [
            [4, 4], [4, 10], [8, 4], [8, 10], [12, 4], [12, 10],
            [6, 3], [6, 11], [9, 3], [9, 11]
        ]
        obstacles.extend(horizontal_obstacles)
        safe_positions = [
            [0, 0], [0, 1], [1, 0], [1, 1],
            [13, 13], [13, 14], [14, 13], [14, 14]
        ]
        obstacles = [obs for obs in obstacles if obs not in safe_positions]
        random_obstacles = [
            [3, 7], [11, 7], [5, 2], [9, 2], [5, 12], [9, 12]
        ]
        obstacles.extend(random_obstacles)
        unique_obstacles = []
        for obs in obstacles:
            if obs not in unique_obstacles:
                unique_obstacles.append(obs)
        return unique_obstacles
    
    def _calculate_low_signal_areas(self):
        low_signal_areas = []
        central_path_low_signal = [
            [7, 6], [7, 7], [7, 8], [8, 6], [8, 7], [8, 8],
            [9, 6], [9, 7], [9, 8]
        ]
        low_signal_areas.extend(central_path_low_signal)
        right_path_low_signal = [
            [11, 6], [11, 7], [11, 8], [12, 6], [12, 7], [12, 8]
        ]
        low_signal_areas.extend(right_path_low_signal)
        left_path_low_signal = [
            [3, 3], [3, 11], [4, 3], [4, 11]
        ]
        low_signal_areas.extend(left_path_low_signal)
        for obs_cell in self.obstacles:
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx = obs_cell[0] + dx
                ny = obs_cell[1] + dy
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 
                    [nx, ny] not in self.obstacles and [nx, ny] not in low_signal_areas):
                    if random.random() < 0.7:
                        low_signal_areas.append([nx, ny])
        unique_areas = []
        safe_distance = 1
        for area in low_signal_areas:
            start_dist = abs(area[0] - self.start_pos[0]) + abs(area[1] - self.start_pos[1])
            end_dist = abs(area[0] - self.target_pos[0]) + abs(area[1] - self.target_pos[1])
            if (area not in unique_areas and 
                area not in self.obstacles and 
                start_dist > safe_distance and 
                end_dist > safe_distance):
                unique_areas.append(area)
        return unique_areas
    
    def _get_local_signal_map(self):
        """返回局部栅格特征向量（展平），取值约 0–1。"""
        local_map = np.zeros((self.local_map_size, self.local_map_size), dtype=np.float32)
        
        center = self.local_map_size // 2
        start_x = self.uav_pos[0] - center
        start_y = self.uav_pos[1] - center
        
        for i in range(self.local_map_size):
            for j in range(self.local_map_size):
                global_x = start_x + i
                global_y = start_y + j
                if 0 <= global_x < self.grid_size and 0 <= global_y < self.grid_size:
                    pos = [global_x, global_y]
                    if self._is_collision(pos):
                        local_map[j, i] = 0.0
                    else:
                        local_map[j, i] = 1.0
                    if global_x == self.target_pos[0] and global_y == self.target_pos[1]:
                        local_map[j, i] = 0.7
                    if global_x == self.start_pos[0] and global_y == self.start_pos[1]:
                        local_map[j, i] = 0.6
                else:
                    local_map[j, i] = 0.0
        local_map[center, center] = 0.9
        return local_map.flatten()

    def _get_observation(self):
        if self.use_local_map:
            local_map = self._get_local_signal_map()
            dx = self.target_pos[0] - self.uav_pos[0]
            dy = self.target_pos[1] - self.uav_pos[1]
            distance = np.sqrt(dx*dx + dy*dy)
            signal_strength = self._get_signal_strength()
            steps_remaining = (self.max_steps - self.steps_taken) / self.max_steps
            
            global_info = np.array([dx, dy, distance, signal_strength, steps_remaining, self.prev_reward], dtype=np.float32)
            observation = np.concatenate([local_map, global_info])
        else:
            dx = self.target_pos[0] - self.uav_pos[0]
            dy = self.target_pos[1] - self.uav_pos[1]
            distance = np.sqrt(dx*dx + dy*dy)
            signal_strength = self._get_signal_strength()
            steps_remaining = (self.max_steps - self.steps_taken) / self.max_steps
            
            observation = np.array([dx, dy, distance, signal_strength, steps_remaining, self.prev_reward], dtype=np.float32)
        
        return observation
    
    def reset(self):
        # Reset UAV to starting position
        self.uav_pos = self.start_pos.copy()
        self.prev_pos = self.start_pos.copy()
        self.steps_taken = 0
        self.reached_target = False
        self.prev_reward = 0.0
        self.prev_signal_strength = None
        self.episode_count += 1
        target = self.target_pos
        self.prev_distance = np.sqrt((self.uav_pos[0] - target[0])**2 + 
                                   (self.uav_pos[1] - target[1])**2)
        
        self.initial_distance = self.prev_distance
        self.trajectory = [self.uav_pos.copy()]
        self.visited_positions = {(self.uav_pos[0], self.uav_pos[1])}
        self.state = self._get_observation()
                                   
        return self.state
    
    def step(self, action):
        self.steps_taken += 1
        self.prev_pos = self.uav_pos.copy()
        self.prev_distance = self._get_closest_distance_to_target()
        if action == 0:
            self.uav_pos[1] = min(self.grid_size-1, self.uav_pos[1] + 1)
        elif action == 1:
            self.uav_pos[0] = min(self.grid_size-1, self.uav_pos[0] + 1)
        elif action == 2:
            self.uav_pos[1] = max(0, self.uav_pos[1] - 1)
        elif action == 3:
            self.uav_pos[0] = max(0, self.uav_pos[0] - 1)
        self.trajectory.append(self.uav_pos.copy())
        self.visited_positions.add(tuple(self.uav_pos))
        current_distance = np.sqrt((self.uav_pos[0] - self.target_pos[0])**2 + 
                                 (self.uav_pos[1] - self.target_pos[1])**2)
        self.reached_target = current_distance < 0.5
        
        current_signal = self._get_signal_strength()
        reward = self._calculate_reward(current_signal)
        self.prev_signal_strength = current_signal
        observation = self._get_observation()
        done = self.reached_target or self.steps_taken >= self.max_steps
        info = {
            'distance_to_target': current_distance,
            'reached_target': self.reached_target,
            'steps': self.steps_taken,
            'signal_strength': current_signal
        }
        
        return observation, reward, done, info
    
    def _evaluate_path_discovery(self):
        coverage = len(self.visited_positions) / (self.grid_size * self.grid_size)
        target_direction = [self.target_pos[0] - self.start_pos[0], 
                          self.target_pos[1] - self.start_pos[1]]
        norm = np.sqrt(target_direction[0]**2 + target_direction[1]**2)
        if norm > 0:
            target_direction = [target_direction[0]/norm, target_direction[1]/norm]
        
        direction_consistency = 0
        for i in range(1, len(self.trajectory)):
            movement = [self.trajectory[i][0] - self.trajectory[i-1][0],
                        self.trajectory[i][1] - self.trajectory[i-1][1]]
            if movement[0] != 0 or movement[1] != 0:
                movement_norm = np.sqrt(movement[0]**2 + movement[1]**2)
                normalized_movement = [movement[0]/movement_norm, movement[1]/movement_norm]
                
                alignment = normalized_movement[0]*target_direction[0] + \
                          normalized_movement[1]*target_direction[1]
                
                if alignment > 0:
                    direction_consistency += alignment
        if len(self.trajectory) > 1:
            direction_consistency /= (len(self.trajectory) - 1)
        current_success_rate = self._get_current_success_rate()
        success_factor = 1.0 - current_success_rate
        path_discovery_score = 0.4 * coverage + 0.6 * direction_consistency
        path_discovery_score *= (0.5 + 0.5 * success_factor)
        return path_discovery_score
    
    def _get_closest_distance_to_target(self):
        """获取轨迹中与目标最近的距离"""
        min_distance = float('inf')
        for pos in self.trajectory:
            distance = np.sqrt((pos[0] - self.target_pos[0])**2 + 
                             (pos[1] - self.target_pos[1])**2)
            min_distance = min(min_distance, distance)
        return min_distance

    def _get_progressive_success_reward(self):
        """计算渐进式成功奖励，基于效率和训练进度"""
        # 基础奖励部分 - 随训练进度逐渐增加
        base_reward = 50.0
        
        # 效率奖励 - 步数越少奖励越高
        efficiency_factor = 1.0 - (self.steps_taken / self.max_steps)
        efficiency_bonus = 50.0 * efficiency_factor
        
        # 训练进度因子 - 随着训练进行逐渐增加奖励
        progress_factor = self._get_training_progress_factor()
        progress_bonus = 50.0 * progress_factor
        
        # 成功率调整 - 当成功率高时降低奖励，当成功率低时增加奖励
        success_rate = self._get_current_success_rate()
        success_rate_factor = 1.0 - success_rate  # 成功率低时提高奖励
        success_rate_bonus = 20.0 * success_rate_factor
        
        # 路径独特性奖励 - 奖励新发现的路径
        path_uniqueness = self._evaluate_path_discovery()
        path_bonus = 30.0 * path_uniqueness
        
        # 总成功奖励
        total_success_reward = base_reward + efficiency_bonus + progress_bonus + success_rate_bonus + path_bonus
        
        return total_success_reward
    
    def _get_training_progress_factor(self):
        """获取训练进度因子，范围从0.5到1.0"""
        if self.total_episodes <= 1:
            return 0.75  # 默认中间值
        
        # 将进度归一化到0.5-1.0范围
        progress = min(1.0, self.episode_count / self.total_episodes)
        return 0.5 + (0.5 * progress)
    
    def _get_current_success_rate(self):
        """计算当前成功率"""
        if not self.success_history:
            return 0.0
        return sum(self.success_history) / len(self.success_history)
    
    def _get_collision_penalty(self):
        """获取碰撞惩罚，随训练进度增加"""
        base_penalty = 10.0
        progress_factor = self._get_training_progress_factor()
        return base_penalty + (progress_factor * 10.0)  # 10-20之间动态调整
    
    def _is_collision(self, pos):
        """Check if position is on an obstacle"""
        for obstacle_cell in self.obstacles:
            if pos[0] == obstacle_cell[0] and pos[1] == obstacle_cell[1]:
                return True
        return False
    
    def _is_in_low_signal_area(self, pos):
        """Check if position is in a low signal area"""
        for area in self.low_signal_areas:
            if pos[0] == area[0] and pos[1] == area[1]:
                return True
        return False
    
    def _get_signal_strength(self):
        distance_to_base = np.sqrt(
            (self.uav_pos[0] - self.start_pos[0])**2 + 
            (self.uav_pos[1] - self.start_pos[1])**2
        )
        distance_factor = np.exp(-0.25 * distance_to_base)
        obstacle_penalty = 0.0
        for obstacle in self.obstacles:
            obs_distance = np.sqrt(
                (self.uav_pos[0] - obstacle[0])**2 + 
                (self.uav_pos[1] - obstacle[1])**2
            )
            if obs_distance <= 2.5:
                obstacle_penalty += 0.4 * (2.5 - obs_distance) / 2.5
        
        signal_strength = distance_factor - obstacle_penalty
        signal_strength = max(0.01, min(1.0, signal_strength))
        return signal_strength
    
    def _calculate_reward(self, current_signal=None):
        reward = 0
        if self.reached_target:
            steps_bonus = max(0, (self.max_steps - self.steps_taken) / self.max_steps) * 100
            reward = 800.0 + steps_bonus
            return reward
        current_distance = self._get_closest_distance_to_target()
        distance_improvement = self.prev_distance - current_distance
        reward += 15.0 * distance_improvement
        max_dist = np.sqrt((self.start_pos[0] - self.target_pos[0])**2 + (self.start_pos[1] - self.target_pos[1])**2)
        dist_to_target = np.sqrt((self.uav_pos[0] - self.target_pos[0])**2 + (self.uav_pos[1] - self.target_pos[1])**2)
        signal_weight = max(0.2, 1.0 - dist_to_target / max_dist)
        reward += 3.0 * (current_signal if current_signal is not None else self._get_signal_strength()) * signal_weight
        if self.prev_signal_strength is not None and current_signal is not None:
            trend = current_signal - self.prev_signal_strength
            reward += 1.5 * trend
        if self._is_collision(self.uav_pos):
            reward -= 40.0
        if self._is_in_low_signal_area(self.uav_pos):
            reward -= 3.0
        reward -= 0.5
        if self.steps_taken >= self.max_steps:
            reward -= 40.0
        return reward
    
    def render(self, mode='human'):
        plt.figure(figsize=(10, 10))
        plt.xlim([-0.5, self.grid_size - 0.5])
        plt.ylim([-0.5, self.grid_size - 0.5])
        
        for i in range(self.grid_size):
            plt.axhline(i - 0.5, color='lightgray')
            plt.axvline(i - 0.5, color='lightgray')
        
        # Draw low signal areas
        for area in self.low_signal_areas:
            plt.gca().add_patch(patches.Rectangle(
                (area[0] - 0.5, area[1] - 0.5),
                1, 1, fill=True, color='purple', alpha=0.3, label='Low Signal Area'
            ))
        
        # Draw obstacles
        for obs_cell in self.obstacles:
            plt.gca().add_patch(patches.Rectangle(
                (obs_cell[0] - 0.5, obs_cell[1] - 0.5),
                1, 1, fill=True, color='black', label='Obstacle'
            ))
        
        # Draw start and target
        plt.scatter(self.start_pos[0], self.start_pos[1], color='green', s=150, marker='o', label='Start')
        plt.scatter(self.target_pos[0], self.target_pos[1], color='red', s=150, marker='x', label='Target')
        
        # Draw UAV
        plt.scatter(self.uav_pos[0], self.uav_pos[1], color='orange', s=200, marker='*', label='UAV')
        
        success_rate = self._get_current_success_rate()
        plt.title(f'UAV Navigation - Step: {self.steps_taken}, Episode: {self.episode_count} (Success Rate: {success_rate:.2f})')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.grid(True)
        plt.savefig(f'uav_step_{self.steps_taken}.png')
        plt.close()

    def close(self):
        plt.close() 