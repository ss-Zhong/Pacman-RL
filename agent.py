import torch
import os
import random
import utils
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import threading
import asyncio
import numpy as np
from threading import Lock

class ExperienceReplay:
    def __init__(self, max_size=10000):
        self.memory = deque(maxlen=max_size)
        self.lock = Lock()  # 创建锁

    def store(self, state, action, reward, next_state, done):
        with self.lock:  # 加锁
            self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        with self.lock:  # 加锁
            return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        with self.lock:  # 加锁
            return len(self.memory)

class DQN(nn.Module):
    '''
    Optimized DQN model class:
        - Contains convolution layers and fully connected layers.
        - Returns Q-values for each of the 4 actions (L, U, R, D).
    '''
    def __init__(self, action_size = 4):
        super(DQN, self).__init__()

        self.action_size = action_size

        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        # 将x展开为 (batch_size, 2, 31, 28)
        while x.dim() < 4:
            x = x.unsqueeze(0)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # 将数据展平
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def clone(self):
        # 克隆当前模型
        cloned_model = DQN(self.action_size)
        cloned_model.load_state_dict(self.state_dict())
        return cloned_model

class Worker(threading.Thread):
    def __init__(self, global_model, optimizer, memory, env, args, device, index):
        super(Worker, self).__init__()
        
        self.env = env
        self.device = device
        self.index = index

        self.optimizer = optimizer
        self.global_model = global_model.to(self.device)
        self.local_model = self.global_model.clone().to(self.device)  # 克隆全局模型

        self.args = args
        self.memory = memory
        self.epsilon = 0.1      # 探索率
        self.step_total = 0
        self.update_local_frequency = 10
        
    async def run(self):
        save_interval = 1000  # 保存模型的间隔，例如每1000步保存一次
        update_count = 0      # 用于跟踪更新次数

        while True:
            # 若游戏未开始或结束，启动游戏
            if self.env.done == True:
                await asyncio.sleep(1)
                await self.env.enter()
                utils.log_message(f"windows-{self.index} done ", self.env.done)
            
            state, pre_life, pre_score= self.env.get_frame()
            
            while not self.env.done:
                action = self.act(state, self.env.get_possible_direction())  # 选择动作
                await self.env.turn(action)
                next_state, current_life, current_score= self.env.get_frame()
                reward = self.env.get_reward(pre_score, current_score, pre_life, current_life, self.env.stage)
                # print("pre_npc: ", pre_npc)
                # print("pre_pacman: ", pre_pacman)
                # print("pre_score: ", pre_score)
                # print("pre_life:", pre_life)
                # print("current_npc:", current_npc)
                # print("current_pacman:", current_pacman)
                # print("current_score:", current_score)
                # print("current_life", current_life)

                # print("Reward: ", reward)
                # pre_npc = current_npc
                # pre_pacman = current_pacman
                pre_life = current_life
                pre_score = current_score
                # i = 0
                # while True:
                #     next_state, reward, current_npc, current_pacman = self.env.get_frame()
                #     print(i)
                #     print("current_npc:", current_npc)
                #     print("current_pacman:", current_pacman)
                #     print("pre_npc: ", pre_npc)
                #     print("pre_pacman: ", pre_pacman)
                #     # 检查 NPC 位置是否发生变化
                #     if self.has_npc_moved(pre_npc, current_npc):
                #         pre_npc = current_npc  # 更新 NPC 位置
                #         pre_pacman = current_pacman
                #         print("hello")
                #         break  # 检测到新帧，跳出循环
                #     await asyncio.sleep(0.1)  # 等待状态稳定
                #     i = i + 1
                utils.log_message(f"Index: {self.index}, Reward: {reward}")
                
                # 存储经验
                self.store_experience(state, action, reward, next_state, self.env.done)
                
                if len(self.memory) > self.args.batch_size:
                    # 进行异步更新
                    await self.update_global_model()

                    update_count += 1
                    # 定期保存模型
                    if update_count % save_interval == 0:
                        await self.save_model(update_count=update_count)

                state = next_state
            print("Done")

    async def save_model(self, update_count):
        model_path = f"model_{self.index}_step_{update_count}.pth"
        torch.save(self.global_model.state_dict(), model_path)
        utils.log_message(f"Model saved at {model_path}")
        print(f"Model saved at {os.path.abspath(model_path)}")

    def has_npc_moved(self, pre_npc, current_npc):
        for pre, current in zip(pre_npc, current_npc):
            if pre != current:  # 如果任何一个NPC的坐标不相等
                return True
        return False
    
    def act(self, state, dircs):
        if random.random() < self.epsilon:
            return random.choice(dircs)  # 探索
        
        with torch.no_grad():
            state = state.to(self.device)
            q_values = self.local_model.forward(state)  # 获取所有 Q 值
            # 过滤 Q 值，保留 dircs 中的索引
            filtered_q_values = [q_values[0][i].item() for i in dircs]
            max_value_index = filtered_q_values.index(max(filtered_q_values))  # 找到最大值的索引

            return dircs[max_value_index]  # 返回对应的动作

    async def update_global_model(self):
        self.step_total += 1
        if self.step_total % self.update_local_frequency == 0: # 定期更新本地模型
            self.update_local_model()

        # 从本地模型中获取经验并更新全局模型
        loss = self.compute_loss()
        
        # 更新全局模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_epsilon()

        if self.step_total % 10 == 0:
            if self.step_total != 0:
                self.step_total = 0

            utils.log_message(f"Index: {self.index}, Loss: {loss.item()}, Epsilon: {self.epsilon}")

    def update_local_model(self):
        self.local_model.load_state_dict(self.global_model.state_dict())

    def compute_loss(self, gamma=0.99):
        experiences = self.memory.sample(self.args.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 计算 Q 值
        q_values = self.global_model(states).gather(1, actions)

        next_q_values = self.local_model(next_states).max(1)[0].detach()  # 使用过去的，增强稳定性
        target_q_values = rewards + (1 - dones) * gamma * next_q_values

        # 计算损失
        loss = F.mse_loss(q_values, target_q_values)

        return loss
    
    def update_epsilon(self):       
        if self.epsilon > 0.0001:
            self.epsilon -= 1e-5

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)