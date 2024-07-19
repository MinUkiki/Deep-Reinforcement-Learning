import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.size < self.capacity:
            self.size += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

class Memory:
    def __init__(self, capacity, alpha):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.e = 0.01

    def _get_priority(self, error):
        return (error + self.e) ** self.alpha

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n
        priorities = []
        indices = []
        for i in range(n):
            s = random.uniform(segment * i, segment * (i + 1))
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            priorities.append(p)
            indices.append(idx)
        return batch, indices, priorities

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, min_epsilon, buffer_size, batch_size, alpha, beta):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001

        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.memory = Memory(buffer_size, alpha)
        
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_values = self.q_network(state)
            action = torch.argmax(action_values).item()
        change_action = -2 + (action / (self.action_dim - 1)) * 4
        return change_action, action

    def remember(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        with torch.no_grad():
            target = reward + self.gamma * self.target_network(next_state).max(1)[0] * (1 - done)
            current = self.q_network(state).gather(1, torch.tensor([[action]]).to(device)).squeeze(1)
            error = abs(target - current).cpu().data.numpy()
        self.memory.add(error, (state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done))

    def replay(self):
        if len(self.memory.tree.data) < self.batch_size:
            return

        batch, idxs, priorities = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).squeeze(1).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).squeeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        q_values = self.q_network(states).gather(1, actions)
        next_action = self.q_network(next_states).max(1)[1].unsqueeze(1)
        q_hat_values = self.target_network(next_states).gather(1, next_action).detach()
        target_q_values = rewards + (self.gamma * q_hat_values * (1 - dones))

        # Importance 샘플링 가중치 계산
        priorities = torch.FloatTensor(priorities).to(device)
        sampling_probabilities = priorities / self.memory.tree.total()
        is_weight = torch.pow(len(self.memory.tree.data) * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        loss = (is_weight * (q_values - target_q_values) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for i in range(len(batch)):
            state, action, reward, next_state, done = batch[i]
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            with torch.no_grad():
                target = reward + self.gamma * self.target_network(next_state).max(1)[0] * (1 - done)
                current = self.q_network(state).gather(1, torch.tensor([[action]]).to(device)).squeeze(1)
                error = abs(target - current).cpu().data.numpy()
            self.memory.update(idxs[i], error)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Hyperparameters
episodes = 500
lr = 0.01
gamma = 0.98
epsilon = 1.0
epsilon_decay = 0.98
min_epsilon = 0.001
buffer_size = 100000
batch_size = 64
update_target_frequency = 10
alpha = 0.6
beta = 0.4

env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] * 11

agent = DQNAgent(state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, min_epsilon, buffer_size, batch_size, alpha, beta)

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

    if episode % update_target_frequency == 0:
        agent.update_target_network()

    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

# Save the trained model
torch.save(agent.q_network.state_dict(), 'dqn_pendulum.pth')
