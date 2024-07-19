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
        action_dist = torch.distributions.Normal(action_mean, torch.tensor([0.1]).to(device))
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def update(self, trajectories):
        states = torch.FloatTensor([t[0] for t in trajectories]).to(device)
        actions = torch.FloatTensor([t[1] for t in trajectories]).to(device)
        rewards = [t[2] for t in trajectories]
        log_probs = torch.cat([t[3] for t in trajectories]).to(device)

        returns = self.compute_returns(rewards)
        returns = torch.FloatTensor(returns).to(device)

        values = self.value_net(states).squeeze()
        deltas = returns - values

        policy_loss = -(deltas.detach() * log_probs).mean()
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
        action, log_prob = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step([action])
        trajectory.append((state, action, reward, log_prob))
        state = next_state
        total_reward += reward

        if done or truncated:
            break

    agent.update(trajectory)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

print("Training completed.")
