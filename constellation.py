import numpy as np
from collections import deque
from satellite import Satellite
import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

state_mapping = {
    'low': [0],
    'medium': [1],
    'high': [2]
}

class Constellation:
    MAX_ITERATIONS = 300
    iteration_count = 0

  # 新增DQN相关属性
    REPLAY_MEMORY_SIZE = 10000
    BATCH_SIZE = 1280
    GAMMA = 0.95
    EPSILON = 0.1
    EPSILON_DECAY = 0.995
    EPSILON_MIN = 0.01
    TARGET_UPDATE = 200

    def __init__(self):
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.dqn_model = None
        self.target_dqn_model = None
        self.optimizer = None
        
    def reset_dqn(self, start_satellite, end_satellite):
   
        # 重置所有卫星的连接数
        for sat in self.satellites:
            sat.num_connections = 0
        
        # 返回起始卫星的初始状态
        return start_satellite.get_state(end_satellite.index)   
        
    def train_dqn(self, satellites, start_index, end_index):
        self.precompute_matrices(satellites)
        start_satellite = satellites[start_index]
        end_satellite = satellites[end_index]

        # 初始化 DQN 模型
        input_dim = len(self.convert_state_to_vector(start_satellite.get_state(end_satellite.index)))
        
        # print(f"Input dimension: {input_dim}")  # 调试信息
        output_dim = len(satellites)  # 固定输出维度为所有卫星的数量
        #print(f"Output dimension: {output_dim}")  # 调试信息
        self.dqn_model = DQN(input_dim, output_dim)
        # 复制相同的参数来初始化目标网络Q
        self.target_dqn_model = DQN(input_dim, output_dim)
        self.target_dqn_model.load_state_dict(self.dqn_model.state_dict())
        self.optimizer = optim.Adam(self.dqn_model.parameters(), lr=0.001)

        print("Starting DQN Training:")
        for i in range(self.MAX_ITERATIONS):
            print(f"\t{i + 1}/{self.MAX_ITERATIONS}")
            self.iteration_count = i + 1
            optimal_path = self.train_dqn_iteration(start_satellite, end_satellite)
            

        print("Training complete, DQN optimal path:", [sat.index for sat in optimal_path])
        return optimal_path

    def train_dqn_iteration(self, start_satellite, end_satellite):
        
        # 重置环境并获取初始状态
        start_satellite = self.reset_dqn(start_satellite, end_satellite)
        
        current_satellite = start_satellite
        path = [current_satellite]
        max_steps = 10000
        step = 0
        while current_satellite != end_satellite:
            if step > max_steps:
                print(f"Max steps exceeded in iteration.")
                break

            state_current = current_satellite.get_state(end_satellite.index)
            possible_actions = current_satellite.get_possible_actions()
            if not possible_actions:
                # No possible actions; terminate the episode
                break

            if random.random() < self.EPSILON:
                    action_index = random.randint(0, len(possible_actions) - 1)
            else:
                state_vector = self.convert_state_to_vector(state_current)
                #print(f"State vector: {state_vector}")  # 调试信息
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                #print(f"State tensor: {state_tensor}")  # 调试信息
                actions_values = self.dqn_model(state_tensor)             
                
                 # 创建掩码，将不可行的动作对应的 Q 值设置为 -inf
                mask = [1 if sat in possible_actions else -float('inf') for sat in self.satellites]
                masked_actions_values = actions_values[0] + torch.tensor(mask, dtype=torch.float32)
                # print(f"Masked actions values: {masked_actions_values}")  # 调试信息
                action_index = torch.argmax(masked_actions_values).item()
                # print(f"Action index: {action_index}")  # 调试信息
                # print(f"Possible actions: {possible_actions}")  # 调试信息
                
                # 确保 action_index 对应的卫星在 possible_actions 中
                if self.satellites[action_index] not in possible_actions:
                    print(f"Warning: Action index {action_index} corresponds to an unreachable satellite. Using random action.")
                    action_index = random.randint(0, len(possible_actions) - 1)

               
            action_current = self.satellites[action_index]
            next_satellite = action_current

            is_final = next_satellite == end_satellite
            state_next = next_satellite.get_state(end_satellite.index)
            reward = current_satellite.get_reward_qlearning(state_next, is_final)

            # 存储经验到回放缓冲区
            self.replay_memory.append((state_current, action_index, reward, state_next, is_final))

            # 当经验池超过一定数量后，训练网络
            if len(self.replay_memory) >= self.BATCH_SIZE:
                self.train_dqn_from_memory()
                
            # 更新目标网络
            if step % self.TARGET_UPDATE == 0:
                self.target_dqn_model.load_state_dict(self.dqn_model.state_dict())

            # Move to the next satellite
            current_satellite = next_satellite
            path.append(current_satellite)

            step += 1

            if is_final:
                break

        # 衰减探索率
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY

        return path

    def train_dqn_from_memory(self):
        minibatch = random.sample(self.replay_memory, self.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = [self.convert_state_to_vector(state) for state in states]
        next_states = [self.convert_state_to_vector(state) for state in next_states]

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        actions_values = self.dqn_model(states)
        next_actions_values = self.target_dqn_model(next_states)
        max_next_actions_values = torch.max(next_actions_values, dim=1)[0]

        target_actions_values = actions_values.clone()
        for i in range(self.BATCH_SIZE):
            # 确保动作索引在有效范围内
            action_index = actions[i].item()
            if action_index < target_actions_values.shape[1]:
                target_actions_values[i][action_index] = rewards[i] + (1 - dones[i]) * self.GAMMA * max_next_actions_values[i]
            else:
                print(f"Warning: Action index {action_index} is out of bounds for target_actions_values with shape {target_actions_values.shape}. Skipping update.")
                print(f"States shape: {states.shape}, Actions: {actions}, Rewards shape: {rewards.shape}, Next states shape: {next_states.shape}, Dones shape: {dones.shape}")  # 调试信息

        loss = nn.MSELoss()(actions_values, target_actions_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def convert_state_to_vector(self, state):
            vector = []
           # print(f"State: {state}")  # 调试信息
            for s in state:
                # print(f"State element: {s}")  # 调试信息
                # print(f"State mapping: {state_mapping[s]}")  # 调试信息
                vector.extend(state_mapping[s])
            
            # print(f"State: {state}, Vector: {vector}")  # 调试信息
            
            return vector
        
    def precompute_matrices(self, satellites):
        self.satellites = satellites
        num_satellites = len(self.satellites)
        
        Satellite.satellites = satellites
        Satellite.visibility_matrix = np.zeros((num_satellites, num_satellites), dtype=bool)
        Satellite.distance_matrix = np.zeros((num_satellites, num_satellites))
        Satellite.latency_matrix = np.empty((num_satellites, num_satellites), dtype=object)

        # Assign index to each satellite 
        for i, satellite in enumerate(self.satellites):
            satellite.index = i

        # Loop through every satellite pair to pre-compute state
        for a in range(num_satellites):
            for b in range(num_satellites):
                if a == b: # Same satellite
                    Satellite.visibility_matrix[a][b] = False
                    Satellite.distance_matrix[a][b] = 0
                    Satellite.latency_matrix[a][b] = 'low'
                else:

                    sat1 = self.satellites[a]
                    sat2 = self.satellites[b]

                    # Compute visibility
                    Satellite.visibility_matrix[a][b] = not sat1.out_of_sight(sat2)

                    # Compute distance
                    distance = sat1.calculate_distance(sat2)
                    Satellite.distance_matrix[a][b] = distance

                    if distance <= Satellite.DELAY_LOW:
                        latency = 'low'
                    elif distance <= Satellite.DELAY_MEDIUM:
                        latency = 'medium'
                    else:
                        latency = 'high'

                    Satellite.latency_matrix[a][b] = latency

    def train_qlearning_iteration(self, start_satellite, end_satellite):
        current_satellite = start_satellite
        path = [current_satellite]
        max_steps = 10000
        step = 0
        while current_satellite != end_satellite:
            if step > max_steps:
                print(f"Max steps exceeded in iteration.")
                break

            state_current = current_satellite.get_state(end_satellite.index)
            possible_actions = current_satellite.get_possible_actions()
            if not possible_actions:
                # No possible actions; terminate the episode
                break

            action_current = current_satellite.qlearning_choose_action(
                state_current, possible_actions
            )
            next_satellite = action_current

            # Simulate adding a connection (increasing congestion)
            # current_satellite.num_connections += 1
            # next_satellite.num_connections += 1

            is_final = next_satellite == end_satellite
            state_next = next_satellite.get_state(end_satellite.index)
            reward = current_satellite.get_reward_qlearning(state_next, is_final)

            current_satellite.update_q_value(
                state_current, action_current, reward, state_next
            )

            # Simulate removing the connection (decreasing congestion)
            # current_satellite.num_connections -= 1
            # next_satellite.num_connections -= 1

            # Move to the next satellite
            current_satellite = next_satellite
            path.append(current_satellite)

            step +=1

            if is_final:
                break
        return path

    def train_qlearning(self, satellites, start_index, end_index):
        self.precompute_matrices(satellites)
        start_satellite = self.satellites[start_index]
        end_satellite = self.satellites[end_index]

        print("Starting Q-Learning Training:")
        for i in range(self.MAX_ITERATIONS):
            #print(f"\t{i+1}/{self.MAX_ITERATIONS}")
            self.iteration_count = i + 1
            # Reset connections for all satellites
            # for sat in self.satellites:
            #     sat.num_connections = 0
            
            # Train for one episode
            optimal_path = self.train_qlearning_iteration(start_satellite, end_satellite)

        print("Training complete, qlearning optimal path:", [sat.index for sat in optimal_path])
        return optimal_path

    def train_wrapper(self, satellites, start_index, end_index, results):
        try:
            optimal_path = self.train_qlearning(satellites, start_index, end_index)
            results.put(optimal_path)
        except Exception as e:
            if e.errno == errno.EPIPE: 
                pass
            else:
                print("error", str(e))

    def flood(self, satellites, start_index, end_index):
        connections = []  # To store the connections formed during flooding
        visited = set()    # To keep track of satellites that have already sent the signal
        queue = deque()

        # Initialize the flood sequence from start satellite
        self.precompute_matrices(satellites)
        queue.append(start_index)
        visited.add(start_index)

        while queue:
            current_index = queue.popleft()
            neighbouring_satellites = [sat.index for sat in self.satellites[current_index].get_possible_actions()]

            for next_index in neighbouring_satellites:
                if next_index not in visited:
                    connections.append([self.satellites[current_index], self.satellites[next_index]]) # Keep track of all connections
                    queue.append(next_index) # Add neighbouring sats to the queue
                    visited.add(next_index)

                    # Check if the end_satellite has been reached
                    if next_index == end_index:
                        return connections

        return connections


    def compare_routing_methods(self, satellites, start_index=None, end_index=None, mas_optimized_path=[], flood_optimized_path=[], dqn_optimized_path=[]):
        # MAS-optimized Path using Q-Learning
        if(flood_optimized_path == []): # If a path is passed in then don't re-calculate path
            mas_optimized_path = self.train_qlearning(satellites=satellites, start_index=start_index, end_index=end_index)
        else:
            start_index = mas_optimized_path[0].index
            end_index = mas_optimized_path[-1].index

        # Non-optimized Path using Flooding
        if(flood_optimized_path == []): # If a path is passed in then don't re-calculate path
            flood_optimized_path = self.flood(satellites=satellites, start_index=start_index, end_index=end_index)

        # DQN-optimized Path
        dqn_optimized_path = self.train_dqn(satellites=satellites, start_index=start_index, end_index=end_index)

        # Initialize congestion counts
        flooding_congestion_counts = {"low": 0, "medium": 0, "high": 0}
        qlearning_congestion_counts = {"low": 0, "medium": 0, "high": 0}
        dqn_congestion_counts = {"low": 0, "medium": 0, "high": 0}

        # Initialize delay counts
        flooding_delay_counts = {"low": 0, "medium": 0, "high": 0}
        qlearning_delay_counts = {"low": 0, "medium": 0, "high": 0}
        dqn_delay_counts = {"low": 0, "medium": 0, "high": 0}

        # Count congestion for the flooding algorithm path
        for connection in flood_optimized_path:
            sat1, sat2 = connection
            flooding_congestion_counts[sat1.check_congestion()] += 1
            flooding_congestion_counts[sat2.check_congestion()] += 1
        
        # Count congestion for the qlearning routing algorithm path
        for i in range(len(mas_optimized_path) - 1):
            sat1 = mas_optimized_path[i]
            sat2 = mas_optimized_path[i + 1]
            qlearning_congestion_counts[sat1.check_congestion()] += 1
            qlearning_congestion_counts[sat2.check_congestion()] += 1

        # Count congestion for the DQN routing algorithm path
        for i in range(len(dqn_optimized_path) - 1):
            sat1 = dqn_optimized_path[i]
            sat2 = dqn_optimized_path[i + 1]
            dqn_congestion_counts[sat1.check_congestion()] += 1
            dqn_congestion_counts[sat2.check_congestion()] += 1

        # Count delay for the flooding algorithm path
        for connection in flood_optimized_path:
            sat1, sat2 = connection
            delay_state1 = sat1.check_latency(sat2)
            delay_state2 = sat2.check_latency(sat1)
            flooding_delay_counts[delay_state1] += 1
            flooding_delay_counts[delay_state2] += 1
        
        # Count delay for the qlearning routing algorithm path
        for i in range(len(mas_optimized_path) - 1):
            sat1 = mas_optimized_path[i]
            sat2 = mas_optimized_path[i + 1]
            delay_state1 = sat1.check_latency(sat2)
            delay_state2 = sat2.check_latency(sat1)
            qlearning_delay_counts[delay_state1] += 1
            qlearning_delay_counts[delay_state2] += 1

        # Count delay for the DQN routing algorithm path
        for i in range(len(dqn_optimized_path) - 1):
            sat1 = dqn_optimized_path[i]
            sat2 = dqn_optimized_path[i + 1]
            delay_state1 = sat1.check_latency(sat2)
            delay_state2 = sat2.check_latency(sat1)
            dqn_delay_counts[delay_state1] += 1
            dqn_delay_counts[delay_state2] += 1


        mas_optimized_stats = {
            'path': [sat.index for sat in mas_optimized_path],
            'distance': 0,
            'num_satellites': len(mas_optimized_path),
            'true_distance' : Satellite.distance_matrix[start_index][end_index],
            'number_of_congested_satellites': qlearning_congestion_counts,
            'number_of_delayed_satellites': qlearning_delay_counts,
        }

        non_optimized_stats = {
            'path': [[sat[0].index, sat[1].index] for sat in flood_optimized_path],
            'distance': 0,
            'num_satellites': len(flood_optimized_path),
            'true_distance' : Satellite.distance_matrix[start_index][end_index],
            'number_of_congested_satellites': flooding_congestion_counts,
            'number_of_delayed_satellites': flooding_delay_counts,
        }

        dqn_optimized_stats = {
            'path': [sat.index for sat in dqn_optimized_path],
            'distance': 0,
            'num_satellites': len(dqn_optimized_path),
            'true_distance' : Satellite.distance_matrix[start_index][end_index],
            'number_of_congested_satellites': dqn_congestion_counts,
            'number_of_delayed_satellites': dqn_delay_counts,
        }

        # Calculate total distance for MAS-optimized route
        if len(mas_optimized_path) > 1:
            for i in range(len(mas_optimized_path) - 1):
                a = mas_optimized_path[i].index
                b = mas_optimized_path[i+1].index
                mas_optimized_stats['distance'] += Satellite.distance_matrix[a][b]

        # Calculate total distance for non-optimized route
        if len(flood_optimized_path) > 1:
            for i in range(len(flood_optimized_path) - 1):
                a = flood_optimized_path[i][0].index
                b = flood_optimized_path[i][1].index
                non_optimized_stats['distance'] += Satellite.distance_matrix[a][b]

        # Calculate total distance for DQN-optimized route
        if len(dqn_optimized_path) > 1:
            for i in range(len(dqn_optimized_path) - 1):
                a = dqn_optimized_path[i].index
                b = dqn_optimized_path[i+1].index
                dqn_optimized_stats['distance'] += Satellite.distance_matrix[a][b]

        return {"qlearing": mas_optimized_stats, "flood": non_optimized_stats, "dqn": dqn_optimized_stats}

    def compare_routing_methods_qlearning(self, satellites, start_index=None, end_index=None, mas_optimized_path=[], flood_optimized_path=[]):
        # MAS-optimized Path using Q-Learning
        if(flood_optimized_path == []): # If a path is passed in then don't re-calculate path
            mas_optimized_path = self.train_qlearning(satellites=satellites, start_index=start_index, end_index=end_index)
        else:
            start_index = mas_optimized_path[0].index
            end_index = mas_optimized_path[-1].index

        # Non-optimized Path using Flooding
        if(flood_optimized_path == []): # If a path is passed in then don't re-calculate path
            flood_optimized_path = self.flood(satellites=satellites, start_index=start_index, end_index=end_index)

        # Initialize congestion counts
        flooding_congestion_counts = {"low": 0, "medium": 0, "high": 0}
        qlearning_congestion_counts = {"low": 0, "medium": 0, "high": 0}

        # Initialize delay counts
        flooding_delay_counts = {"low": 0, "medium": 0, "high": 0}
        qlearning_delay_counts = {"low": 0, "medium": 0, "high": 0}

        # Count congestion for the flooding algorithm path
        for connection in flood_optimized_path:
            sat1, sat2 = connection
            flooding_congestion_counts[sat1.check_congestion()] += 1
            flooding_congestion_counts[sat2.check_congestion()] += 1
        
        # Count congestion for the qlearning routing algorithm path
        for i in range(len(mas_optimized_path) - 1):
            sat1 = mas_optimized_path[i]
            sat2 = mas_optimized_path[i + 1]
            qlearning_congestion_counts[sat1.check_congestion()] += 1
            qlearning_congestion_counts[sat2.check_congestion()] += 1

        # Count delay for the flooding algorithm path
        for connection in flood_optimized_path:
            sat1, sat2 = connection
            delay_state1 = sat1.check_latency(sat2)
            delay_state2 = sat2.check_latency(sat1)
            flooding_delay_counts[delay_state1] += 1
            flooding_delay_counts[delay_state2] += 1
        
        # Count delay for the qlearning routing algorithm path
        for i in range(len(mas_optimized_path) - 1):
            sat1 = mas_optimized_path[i]
            sat2 = mas_optimized_path[i + 1]
            delay_state1 = sat1.check_latency(sat2)
            delay_state2 = sat2.check_latency(sat1)
            qlearning_delay_counts[delay_state1] += 1
            qlearning_delay_counts[delay_state2] += 1


        mas_optimized_stats = {
            'path': [sat.index for sat in mas_optimized_path],
            'distance': 0,
            'num_satellites': len(mas_optimized_path),
            'true_distance' : Satellite.distance_matrix[start_index][end_index],
            'number_of_congested_satellites': qlearning_congestion_counts,
            'number_of_delayed_satellites': qlearning_delay_counts,
        }

        non_optimized_stats = {
            'path': [[sat[0].index, sat[1].index] for sat in flood_optimized_path],
            'distance': 0,
            'num_satellites': len(flood_optimized_path),
            'true_distance' : Satellite.distance_matrix[start_index][end_index],
            'number_of_congested_satellites': flooding_congestion_counts,
            'number_of_delayed_satellites': flooding_delay_counts,
        }

        # Calculate total distance for MAS-optimized route
        if len(mas_optimized_path) > 1:
            for i in range(len(mas_optimized_path) - 1):
                a = mas_optimized_path[i].index
                b = mas_optimized_path[i+1].index
                mas_optimized_stats['distance'] += Satellite.distance_matrix[a][b]

        # Calculate total distance for non-optimized route
        if len(flood_optimized_path) > 1:
            for i in range(len(flood_optimized_path) - 1):
                a = flood_optimized_path[i][0].index
                b = flood_optimized_path[i][1].index
                non_optimized_stats['distance'] += Satellite.distance_matrix[a][b]

        return {"qlearning": mas_optimized_stats, "flood": non_optimized_stats}