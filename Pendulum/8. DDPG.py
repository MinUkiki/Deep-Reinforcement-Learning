#import gymnasium as gym # 다른 환경을 사용할 때
import random
import os
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pendulum import PendulumEnv

# 하이퍼파라미터
lr_mu = 0.0001
lr_q = 0.001
gamma = 0.99
batch_size = 64
buffer_limit = 50000
tau = 0.005  # 타겟 네트워크 소프트 업데이트를 위한 파라미터

current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, "saved_model")

# 디렉토리가 없으면 생성
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 리플레이 버퍼
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)

# 액터 네트워크 (정책 네트워크)
class MuNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc_mu = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = torch.tanh(self.fc_mu(x)) * 2
        return mu

# 크리틱 네트워크 (Q 네트워크)
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_dim, 128)
        self.fc_a = nn.Linear(action_dim, 128)
        self.fc_q1 = nn.Linear(256, 128)
        self.fc_q2 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, action_dim)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q1(cat))
        q = F.relu(self.fc_q2(q))
        q = self.fc_out(q)
        return q

# Ornstein-Uhlenbeck 노이즈
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

# 학습 함수
def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s, a, r, s_prime, done_mask = memory.sample(batch_size)

    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.smooth_l1_loss(q(s, a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    mu_loss = -q(s, mu(s)).mean()  # 정책 손실
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()

# 소프트 업데이트 함수
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

# 메인 함수
def main():
    # env = gym.make('Pendulum-v1')
    # env = PendulumEnv(render_mode='human')
    env = PendulumEnv()
    memory = ReplayBuffer()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    q, q_target = QNet(state_dim, action_dim), QNet(state_dim, action_dim)
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(state_dim, action_dim), MuNet(state_dim, action_dim)
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 20

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    for n_epi in range(500):
        s, _ = env.reset()
        done = False

        count = 0
        while count < 200 and not done:
            a = mu(torch.from_numpy(s).float())
            a = a.item() + ou_noise()[0]
            s_prime, r, terminated, truncated, _  = env.step([a])
            done = terminated or truncated
            memory.put((s, a, r, s_prime, done))
            score += r
            s = s_prime
            count += 1

        if memory.size() > 2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q, q_target)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    # 최종 모델 저장
    torch.save(mu.state_dict(), f"{model_dir}/ddpg_actor_final.pth")
    torch.save(q.state_dict(), f"{model_dir}/ddpg_critic_final.pth")
    env.close()

if __name__ == '__main__':
    main()
