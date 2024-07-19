import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * 2  # Pendulum action space is in range [-2, 2]

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma):
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state)
        return action.item()

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        action = torch.FloatTensor([action]).to(device)
        reward = torch.FloatTensor([reward]).to(device)
        done = torch.FloatTensor([done]).to(device)

        value = self.critic(state)
        next_value = self.critic(next_state)
        target = reward + (1 - done) * self.gamma * next_value
        delta = target - value

        actor_loss = -torch.log(action + 1e-10) * delta.detach()
        critic_loss = delta.pow(2)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

# Hyperparameters
lr_actor = 0.001
lr_critic = 0.005
gamma = 0.99
episodes = 1000
max_timesteps = 200

# Environment setup
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = ActorCriticAgent(state_dim, action_dim, lr_actor, lr_critic, gamma)

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(max_timesteps):
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step([action])
        agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done or truncated:
            break

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

print("Training completed.")

'''
주요 구성 요소 설명
액터 네트워크 (Actor Network): 상태를 입력받아 행동을 출력하는 신경망입니다. tanh 활성화 함수를 사용하여 출력이 [-2, 2] 범위에 있도록 합니다.
크리틱 네트워크 (Critic Network): 상태를 입력받아 해당 상태의 가치를 출력하는 신경망입니다.
에이전트 (Agent): 상태를 입력받아 행동을 선택하고, 각 시간 단계마다 정책 및 가치 함수를 업데이트합니다.
환경 (Environment): Pendulum-v1 환경에서 에이전트가 상호작용합니다.
학습 과정 설명
상태와 행동 선택: 에이전트는 현재 상태에서 액터 네트워크를 사용하여 행동을 선택합니다.
행동 수행 및 보상 관찰: 선택한 행동을 환경에 적용하여 다음 상태와 보상을 관찰합니다.
정책 및 가치 함수 업데이트: TD 오류를 계산하고, 이를 바탕으로 액터 및 크리틱 네트워크를 업데이트합니다.
이 코드는 Pendulum-v1 환경에 맞게 작성되었으며, Actor-Critic 알고리즘을 사용하여 정책 및 가치 함수를 학습합니다.
'''