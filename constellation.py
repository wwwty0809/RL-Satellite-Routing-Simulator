import numpy as np
from collections import deque
from satellite import Satellite
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import collections


# --------------------------------------- #
# 经验回放池
# --------------------------------------- #

class ReplayBuffer():
    def __init__(self, capacity):
        # 创建一个先进先出的队列，最大长度为capacity，保证经验池的样本量不变
        self.buffer = collections.deque(maxlen=capacity)
    # 将数据以元组形式添加进经验池
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    # 随机采样batch_size行数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)  # list, len=32
        # *transitions代表取出列表中的值，即32项
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    # 目前队列长度
    def size(self):
        return len(self.buffer)

# -------------------------------------- #
# 构造深度学习网络，输入状态s，得到各个动作的reward
# -------------------------------------- #

class Net(nn.Module):
    # 构造只有一个隐含层的网络
    def __init__(self, n_states, n_hidden, n_actions):
        super(Net, self).__init__()
        # [b,n_states]-->[b,n_hidden]
        self.fc1 = nn.Linear(n_states, n_hidden)
        # [b,n_hidden]-->[b,n_actions]
        self.fc2 = nn.Linear(n_hidden, n_actions)
    # 前传
    def forward(self, x):  # [b,n_states]
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x

# -------------------------------------- #
# 构造深度强化学习模型
# -------------------------------------- #

class DQN:
    #（1）初始化
    def __init__(self, n_states, n_hidden, n_actions,
                 learning_rate, gamma, epsilon,
                 target_update, device):
        #self.satellites = satellites  # 存储卫星列表
        
        # 属性分配
        self.n_states = n_states  # 状态的特征数
        self.n_hidden = n_hidden  # 隐含层个数
        self.n_actions = n_actions  # 动作数
        self.learning_rate = learning_rate  # 训练时的学习率
        self.gamma = gamma  # 折扣因子，对下一状态的回报的缩放
        self.epsilon = epsilon  # 贪婪策略，有1-epsilon的概率探索
        self.target_update = target_update  # 目标网络的参数的更新频率
        self.device = device  # 在GPU计算
        # 计数器，记录迭代次数
        self.count = 0

        # 构建2个神经网络，相同的结构，不同的参数
        # 实例化训练网络  [b,4]-->[b,2]  输出动作对应的奖励
        self.q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        # 实例化目标网络
        self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions)

        # 优化器，更新训练网络的参数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        
        
          #（2）动作选择
    def take_action(self, state_vector, possible_actions, satellites):  # epsilon-贪婪策略采取动作
        # 将状态转换为张量
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)  
        # 如果小于该值就取最大的值对应的索引
        if np.random.random() < self.epsilon:  # 0-1
            # 前向传播获取该状态对应的动作的reward
            actions_value = self.q_net(state_tensor)

                # 创建掩码，将不可行的动作对应的 Q 值设置为 -inf
            mask = [0 if sat in possible_actions else -float('inf') for sat in satellites]
            actions_values = actions_value[0] + torch.tensor(mask, dtype=torch.float32)
           
            # 获取reward最大值对应的动作索引
            action_index = torch.argmax(actions_values).item()
        # 如果大于该值就随机探索
        else:
            # 随机选择一个动作
            action_index = np.random.randint(self.n_actions)
        return action_index

    #（3）网络训练
    def update(self, transition_dict):  # 传入经验池中的batch个样本
        # 获取当前时刻的状态 array_shape=[b,4]
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        # 获取当前时刻采取的动作 tuple_shape=[b]，维度扩充 [b,1]
        actions = torch.tensor(transition_dict['actions']).view(-1,1)
        # 当前状态下采取动作后得到的奖励 tuple=[b]，维度扩充 [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1)
        # 下一时刻的状态 array_shape=[b,4]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        # 是否到达目标 tuple_shape=[b]，维度变换[b,1]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1)

        # 输入当前状态，得到采取各运动得到的奖励 [b,4]==>[b,2]==>[b,1]
        # 根据actions索引在训练网络的输出的第1维度上获取对应索引的q值（state_value）
        q_values = self.q_net(states).gather(1, actions)  # [b,1]
        # 下一时刻的状态[b,4]-->目标网络输出下一时刻对应的动作q值[b,2]-->
        # 选出下个状态采取的动作中最大的q值[b]-->维度调整[b,1]
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)
        # 目标网络输出的当前状态的q(state_value)：即时奖励+折扣因子*下个时刻的最大回报
        q_targets = rewards + self.gamma * max_next_q_values * (1-dones)

        # 目标网络和训练网络之间的均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        # 反向传播参数更新
        dqn_loss.backward()
        # 对训练网络更新
        self.optimizer.step()

        # 在一段时间后更新目标网络的参数
        if self.count % self.target_update == 0:
            # 将目标网络的参数替换成训练网络的参数
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        
        self.count += 1
    
    def convert_state_to_vector(self, state):
            vector = []
           
            for s in state:
                vector.extend(state_mapping[s])
  
            return vector
        
state_mapping = {
    'low': [0],
    'medium': [1],
    'high': [2]
}

# GPU运算
device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
        
# 超参数
capacity = 500  # 经验池容量
lr = 2e-3  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.9  # 贪心系数
target_update = 200  # 目标网络的参数的更新频率
batch_size = 32
n_hidden = 128  # 隐含层神经元个数
min_size = 200  # 经验池超过200后再训练
return_list = []  # 记录每个回合的回报


class Constellation:
    MAX_ITERATIONS = 300
    iteration_count = 0
    def __init__(self, satellites=None):
            """
            初始化星座网络
            :param satellites: 可选的卫星列表，如果为None则创建空列表
            """
            self.satellites = satellites
            self.start_index = None  # 初始化起点卫星
            self.end_index = None    # 初始化终点卫星
            self.precompute_matrices(self.satellites)  # 预计算矩阵
            self.num_satellites = len(self.satellites)  # 卫星数量
        
    def reset_dqn(self):
        # 重置所有卫星的位置（随机经纬度）
        for sat in self.satellites:
            sat.longitude = np.random.uniform(0, 360)   # 经度范围: 0°~360°
            sat.latitude = np.random.uniform(-90, 90)   # 纬度范围: -90°~90°
            sat.num_connections = 0              # 卫星当前的活动连接数量，用于判断卫星的拥塞状态
            sat.height = 0
            sat.speed = 0.5
        
        # 随机选择起点卫星
        self.start_index = int(np.random.uniform(0, self.num_satellites))
        
        # 随机选择终点卫星（必须与起点不同）
        while True:
            self.end_index = int(np.random.uniform(0, self.num_satellites))
            if self.end_index != self.start_index:
                break
        
        # 重新计算卫星间的可见性、距离和延迟矩阵
        self.precompute_matrices(self.satellites)
        
    
        
    def train_dqn(self, satellites, start_index, end_index):
        self.precompute_matrices(satellites)
        start_satellite = satellites[start_index]
        end_satellite = satellites[end_index]
    
        # 实例化经验池
        self.replay_buffer = ReplayBuffer(capacity)

        # 实例化 DQN 模型
        input_dim = len(self.convert_state_to_vector(start_satellite.get_state(end_satellite.index)))
        output_dim = len(satellites)  # 固定输出维度为所有卫星的数量
        self.dqn_model = DQN(n_states=input_dim,
            n_hidden=n_hidden,
            n_actions=output_dim,
            learning_rate=lr,
            gamma=gamma,
            epsilon=epsilon,
            target_update=target_update,
            device=device,)


        print("Starting DQN Training:")
        for i in range(self.MAX_ITERATIONS):
            print(f"\t{i + 1}/{self.MAX_ITERATIONS}")
            self.iteration_count = i + 1
            
            # 重置环境并获取初始状态
            self.reset_dqn()
            print("start",self.start_index)
            optimal_path = self.train_dqn_iteration(start_satellite, end_satellite)
            

        print("Training complete, DQN optimal path:", [sat.index for sat in optimal_path])
        return optimal_path

    def train_dqn_iteration(self, start_satellite, end_satellite):
        

        # 记录每个回合的回报
        episode_return = 0
        current_satellite = start_satellite
        path = [current_satellite]
        max_steps = 10000
        step = 0
        state_vector = current_satellite.get_state_vector_dqn(end_satellite.index)
        while current_satellite != end_satellite:
            if step > max_steps:
                print(f"Max steps exceeded in iteration.")
                break

            possible_actions = current_satellite.get_possible_actions()
            if not possible_actions:
                # No possible actions; terminate the episode
                break
            

            # 获取当前状态下需要采取的动作
            action_index = self.dqn_model.take_action(state_vector, possible_actions, self.satellites)
            action_current = self.satellites[action_index]
            
            if not action_current:
                # No possible actions; terminate the episode
                break
            # 更新环境
            next_satellite = action_current
            is_final = next_satellite == end_satellite
            next_state = next_satellite.get_state_vector_dqn(end_satellite.index)
            #print("next_state:",next_state)
            reward = current_satellite.get_reward_dqn(next_state, is_final)
            
            #next_state, reward, done, _ = env.step(action)
            # 添加经验池
            self.replay_buffer.add(state_vector, action_index, reward, next_state, is_final)
            # 更新当前状态
            state_vector = next_state
            #print("state_vector:",state_vector)
            # 更新回合回报
            episode_return += reward

            # 当经验池超过一定数量后，训练网络
            if self.replay_buffer.size() > min_size:
                # 从经验池中随机抽样作为训练集
                s, a, r, ns, d = self.replay_buffer.sample(batch_size)
                # 构造训练集
                transition_dict = {
                    'states': s,
                    'actions': a,
                    'next_states': ns,
                    'rewards': r,
                    'dones': d,
                }
                
                #print("transition_dict",transition_dict)
                # 网络更新
                self.dqn_model.update(transition_dict)
            # 找到目标就结束
            #if done: break
            current_satellite = next_satellite
            path.append(current_satellite)

            step += 1

            if is_final:
                break


        return path


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
            for sat in self.satellites:
                sat.num_connections = 0
            
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