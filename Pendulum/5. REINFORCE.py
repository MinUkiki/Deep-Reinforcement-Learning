import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, action_dim)
        self.fc_log_std = nn.Linear(32, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))         # 평균 값
        std = F.softplus(self.fc_log_std(x))                    # 표준편차
        return mu, std

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr_policy, lr_value, gamma):
        self.policy_network = PolicyNetwork(state_dim, action_dim).to(device)
        self.value_network = ValueNetwork(state_dim).to(device)
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=lr_value)
        self.gamma = gamma
        self.memory = []

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mu, std = self.policy_network(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def remember(self, log_prob, value, reward):
        self.memory.append((log_prob, value, reward))

    def update(self):
        R = 0
        policy_loss = []
        value_loss = []
        returns = []

        for log_prob, value, reward in reversed(self.memory):
            R = reward + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for (log_prob, value, reward), R in zip(self.memory, returns):
            advantage = R - value.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(nn.MSELoss()(value, torch.tensor([R]).to(device)))

        self.optimizer_policy.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer_policy.step()

        self.optimizer_value.zero_grad()
        value_loss = torch.stack(value_loss).sum()
        value_loss.backward()
        self.optimizer_value.step()

        self.memory = []

# 하이퍼파라미터
episodes = 5000
lr_policy = 0.001
lr_value = 0.001
gamma = 0.99
train_score = []

# 환경 설정
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# 에이전트 생성
agent = REINFORCEAgent(state_dim, action_dim, lr_policy, lr_value, gamma)

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, log_prob = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step([2.0*action]) # [-2,2]
        done = terminated or truncated
        value = agent.value_network(torch.FloatTensor(state).unsqueeze(0).to(device))
        agent.remember(log_prob, value, reward)
        state = next_state
        total_reward += reward

    agent.update()
    train_score.append(total_reward)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# 학습된 모델 저장
torch.save(agent.policy_network.state_dict(), 'Pendulum/save_model/reinforce_policy.pth')
torch.save(agent.value_network.state_dict(), 'Pendulum/save_model/reinforce_value.pth')

# 학습된 score 저장
np.savetxt('Pendulum/score_log/pendulum_score_REINFORCE.txt', train_score)
