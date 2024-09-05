import gymnasium as gym # 다른 환경을 사용할 때
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random, os
from pendulum import PendulumEnv

current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, "saved_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, min_epsilon, buffer_size, batch_size):
        # 상태 차원과 행동 차원 저장
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 하이퍼파라미터 저장
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon # 탐색 비율
        self.epsilon_decay = epsilon_decay # 탐색 비율 감소율
        self.min_epsilon = min_epsilon # 최소 탐색 비율
        self.buffer_size = buffer_size # 리플레이 버퍼 크기
        self.batch_size = batch_size # 학습 배치 크기

        # Q 네트워크와 타겟 네트워크 초기화
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        
        # Adam 옵티마이저로 Q 네트워크의 파라미터를 최적화
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 경험을 저장할 buffer 초기화
        self.memory = deque(maxlen=buffer_size)
        
        # 타겟 네트워크를 Q 네트워크의 가중치로 초기화
        self.update_target_network()

    def update_target_network(self):
        # 타겟 네트워크를 Q 네트워크의 가중치로 업데이트
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_action(self, state):
        # 탐색 비율에 따라 무작위 행동 선택 또는 Q 네트워크를 사용하여 행동 선택
        if np.random.rand() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            # 상태를 텐서로 변환하고 Q 네트워크를 사용하여 행동 값 계산
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_values = self.q_network(state)
            # 가장 높은 Q 값을 가지는 행동 선택
            action = torch.argmax(action_values).item()
        # Action을 [-2, 2] 범위로 변환
        change_action = -2 + (action / (self.action_dim -1)) * 4
        return change_action, action

    def remember(self, state, action, reward, next_state, done):
        # 경험을 리플레이 버퍼에 저장
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        # 리플레이 버퍼가 충분히 채워지지 않았으면 학습하지 않음
        if len(self.memory) < self.batch_size:
            return
        
        # 리플레이 버퍼에서 배치 샘플링
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 샘플링된 데이터를 텐서로 변환
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # 현재 상태에 대한 Q 값 계산
        q_values = self.q_network(states).gather(1, actions)
        
        # 다음 상태에 대한 타겟 Q 값 계산
        q_hat_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * q_hat_values * (1 - dones))

        # 손실 함수 계산 (평균 제곱 오차)
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # 손실 역전파 및 네트워크 파라미터 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        # 탐색 비율을 감소
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Hyperparameters
episodes = 300
lr = 0.001
gamma = 0.98
epsilon = 1.0
epsilon_decay = 0.98
min_epsilon = 0.01
buffer_size = 10000
batch_size = 64
update_target_frequency = 10
train_socre = []

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment
# env = gym.make('Pendulum-v1')
env = PendulumEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] * 11

# Agent
agent = DQNAgent(state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, min_epsilon, buffer_size, batch_size)

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        change_action, action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step([change_action])
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward

    agent.decay_epsilon()
    train_socre.append(total_reward)

    if episode % update_target_frequency == 0:
        agent.update_target_network()
        
    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

# 학습된 모델 저장
torch.save(agent.q_network.state_dict(), f"{model_dir}/dqn_pendulum.pth")