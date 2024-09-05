import gymnasium as gym # 다른 환경을 사용할 때
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from pendulum import PendulumEnv

# 하이퍼파라미터 설정
learning_rate = 0.001
gamma = 0.98
current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, "saved_model")

# 디렉토리가 없으면 생성
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(state_dim, 128)  # Pendulum 환경의 상태 공간 크기는 3
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_mean = nn.Linear(64, action_dim)   # 액션의 표준편차
        self.fc_std = nn.Linear(64, action_dim)   # 액션의 표준편차
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = torch.tanh(self.fc_mean(x)) * 2.0  # Pendulum의 액션 범위는 [-2, 2]
        std = F.softplus(self.fc_std(x))  # 표준편차는 항상 양수여야 하므로 softplus 사용
        return mean, std

    def put_data(self, item):
        self.data.append(item)

    def train_net(self, critic):
        R = 0
        self.optimizer.zero_grad()
        for r, prob, state in self.data[::-1]:
            R = r + gamma * R
            value = critic(state)
            td_error = R - value.item()
            loss = -prob * td_error
            loss.backward()
        self.optimizer.step()
        self.data = []

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc4(x)
        return value

    def train_net(self, data):
        R = 0
        loss_list = []
        self.optimizer.zero_grad()
        for r, _, state in data[::-1]:
            R = r + gamma * R
            value = self.forward(state)
            loss = F.mse_loss(value, torch.tensor([[R]], dtype=torch.float32))
            loss_list.append(loss)
        loss = torch.stack(loss_list).mean()
        loss.backward()
        self.optimizer.step()

def main():
    # env = gym.make('Pendulum-v1')
    env = PendulumEnv() 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = Policy(state_dim,action_dim)
    critic = Critic(state_dim,action_dim)
    score = 0.0
    print_interval = 20

    for n_epi in range(1500):
        s, _ = env.reset()
        done = False

        while not done:
            mean, std = policy(torch.from_numpy(s).float())
            dist = Normal(mean, std)
            a = dist.sample()
            a = torch.clamp(a, -2.0, 2.0)  # Pendulum의 액션 범위 [-2, 2]로 클램핑
            s_prime, r, terminated, truncated, _ = env.step(np.array([a.item()], dtype=np.float32))
            done = terminated or truncated
            policy.put_data((r, dist.log_prob(a), torch.from_numpy(s).float()))
            s = s_prime
            score += r

        # Critic 네트워크 학습
        critic.train_net(policy.data)
        # Policy 네트워크 학습 (Baseline 사용)
        policy.train_net(critic)

        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"# of episode :{n_epi}, avg score : {score / print_interval:.1f}")
            score = 0.0
            
    #         # 모델 저장
    #         torch.save(policy.state_dict(), f"{model_dir}/{n_epi}_policy.pth")
    #         torch.save(critic.state_dict(), f"{model_dir}/{n_epi}_critic.pth")

    # 최종 모델 저장
    torch.save(policy.state_dict(), f"{model_dir}/reinforce_policy_pendulum.pth")
    torch.save(critic.state_dict(), f"{model_dir}/reinforce_critic_pendulum.pth")

    env.close()

if __name__ == '__main__':
    main()
