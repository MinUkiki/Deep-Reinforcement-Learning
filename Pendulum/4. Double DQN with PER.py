import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import gymnasium as gym # 다른 환경을 사용할 때
from collections import deque
from pendulum import PendulumEnv

ACTION_SPACE_SIZE = 11
current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, "saved_model")

# Pendulum 환경 설정
# env = gym.make('Pendulum-v1')
env = PendulumEnv()

# 디렉토리가 없으면 생성
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

action_space = np.linspace(-2.0, 2.0, ACTION_SPACE_SIZE)

# Q-네트워크 정의
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Prioritized Replay Buffer 정의
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, transition, td_error):
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append(tuple(map(np.array, transition)))  # 모든 요소를 numpy array로 변환
        self.priorities.append((abs(td_error) + PRIORITY_EPSILON) ** self.alpha)

    def sample(self, batch_size, beta):
        probabilities = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + PRIORITY_EPSILON) ** self.alpha

# Double DQN 에이전트 정의
class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = PrioritizedReplayBuffer(REPLAY_MEMORY_SIZE, ALPHA)
        self.beta = BETA_START
        self.frame_idx = 0

    def select_action(self, state):
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
                action_idx = q_values.argmax().item()
        else:
            action_idx = random.randint(0, self.action_dim - 1)
        return action_idx

    def update(self):
        if len(self.replay_buffer.buffer) < BATCH_SIZE:
            return

        transitions, indices, weights = self.replay_buffer.sample(BATCH_SIZE, self.beta)
        batch = np.array(transitions, dtype=object)
        states = torch.tensor(np.vstack(batch[:, 0]), dtype=torch.float32)
        actions = torch.tensor(batch[:, 1].astype(np.int64), dtype=torch.long)  # action을 int64로 변환
        rewards = torch.tensor(batch[:, 2].astype(np.float32), dtype=torch.float32)
        next_states = torch.tensor(np.vstack(batch[:, 3]), dtype=torch.float32)
        dones = torch.tensor(batch[:, 4].astype(np.float32), dtype=torch.float32)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_network(next_states).detach()
        next_actions = self.q_network(next_states).argmax(1).detach()
        next_q_value = next_q_values.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
        target = rewards + (1 - dones) * GAMMA * next_q_value

        td_errors = q_values - target
        
        self.replay_buffer.update_priorities(indices, td_errors.detach().numpy())
        
        loss = (torch.tensor(weights, dtype=torch.float32) * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.frame_idx % TARGET_UPDATE_FREQ == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.frame_idx += 1

# Hyperparameters
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 10000
GAMMA = 0.99
ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000
PRIORITY_EPSILON = 1e-5
LEARNING_RATE = 3e-4
TARGET_UPDATE_FREQ = 10

# 에이전트 학습
agent = DoubleDQNAgent(state_dim=env.observation_space.shape[0], action_dim=ACTION_SPACE_SIZE)
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 500
print_interval = 10  # 에피소드마다 평균 점수를 출력할 간격
score = 0.0

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action_idx = agent.select_action(state)
        action = np.array([action_space[action_idx]])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 우선순위 리플레이 버퍼에 추가
        target = reward + GAMMA * np.max(agent.q_network(torch.tensor(np.array([next_state]), dtype=torch.float32)).detach().numpy())
        current = np.max(agent.q_network(torch.tensor(np.array([state]), dtype=torch.float32)).detach().numpy())
        td_error = target - current
        agent.replay_buffer.add((state, action_idx, reward, next_state, done), td_error)

        agent.update()

        state = next_state
        total_reward += reward

    score += total_reward
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % print_interval == 0 and episode != 0:
        avg_score = score / print_interval
        print(f"# of episode : {episode}, avg score : {avg_score:.1f}, epsilon : {epsilon}")
        score = 0.0

# 학습이 완료된 후 최종 모델 저장
torch.save(agent.q_network.state_dict(), f"{model_dir}/double_dqn_pendulum.pth")

env.close()
