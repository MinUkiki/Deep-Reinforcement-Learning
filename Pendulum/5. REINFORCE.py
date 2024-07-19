import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Pendulum action space is in range [-2, 2]

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class REINFORCEWithBaselineAgent:
    def __init__(self, state_dim, action_dim, lr_policy, lr_value, gamma):
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_mean = self.policy_net(state)
        action = torch.clamp(action_mean, -2, 2)  # Pendulum action space is in range [-2, 2]
        return action.item()

    def update(self, trajectories):
        states = torch.FloatTensor([t[0] for t in trajectories]).to(device)
        actions = torch.FloatTensor([t[1] for t in trajectories]).to(device)
        rewards = [t[2] for t in trajectories]

        returns = self.compute_returns(rewards)
        returns = torch.FloatTensor(returns).to(device)

        values = self.value_net(states).squeeze()
        deltas = returns - values

        policy_loss = -(deltas.detach() * torch.log(torch.clamp(actions, -2, 2))).mean()
        value_loss = deltas.pow(2).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

# Hyperparameters
lr_policy = 0.001
lr_value = 0.001
gamma = 0.99
episodes = 1000
max_timesteps = 200

# Environment setup
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = REINFORCEWithBaselineAgent(state_dim, action_dim, lr_policy, lr_value, gamma)

for episode in range(episodes):
    state, _ = env.reset()
    trajectory = []
    total_reward = 0

    for t in range(max_timesteps):
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step([action])
        trajectory.append((state, action, reward))
        state = next_state
        total_reward += reward

        if done or truncated:
            break

    agent.update(trajectory)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

print("Training completed.")

'''
주요 구성 요소 설명
정책 네트워크 (Policy Network): 상태를 입력받아 행동을 출력하는 신경망입니다. tanh 활성화 함수를 사용하여 출력이 [-2, 2] 범위에 있도록 합니다.
가치 네트워크 (Value Network): 상태를 입력받아 해당 상태의 가치를 출력하는 신경망입니다.
에이전트 (Agent): 상태를 입력받아 행동을 선택하고, 에피소드 종료 후 학습을 수행합니다.
환경 (Environment): Pendulum-v1 환경에서 에이전트가 상호작용합니다.
학습 과정 설명
에피소드 수집: 에이전트는 환경에서 에피소드를 수행하며 상태, 행동, 보상을 수집합니다.
리턴 계산: 수집된 보상을 바탕으로 리턴을 계산합니다.
정책 및 가치 함수 업데이트: 리턴과 가치 네트워크의 출력을 이용하여 정책 및 가치 함수를 업데이트합니다.
탐험 감소: 에피소드가 진행됨에 따라 탐험 비율을 줄여나갑니다.
이 코드는 Pendulum-v1 환경에 맞게 작성되었으며, REINFORCE with baseline 알고리즘을 사용하여 정책 및 가치 함수를 학습합니다.
'''