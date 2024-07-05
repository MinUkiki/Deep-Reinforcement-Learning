import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, model_path):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.q_network.load_state_dict(torch.load(model_path))
        self.q_network.eval()

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = self.q_network(state)
        return torch.argmax(action_values).item()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] * 11

# Agent
agent = DQNAgent(state_dim, action_dim, 'dqn_pendulum2.pth')

episodes = 10
total_rewards = []

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        env.render()
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step([action])
        if terminated==True or truncated==True:
            done = True
        state = next_state
        total_reward += reward

    total_rewards.append(total_reward)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

env.close()

# 전체 테스트 결과 출력
print("Testing completed")
print(f"Average reward over {episodes} episodes: {np.mean(total_rewards)}")
