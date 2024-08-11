import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from torch.distributions import Normal

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
n_rollout = 10
model_dir = "saved_model"

# 디렉토리가 없으면 생성
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(state_dim, 256)  # Pendulum의 상태 공간 크기는 3
        self.fc_mu = nn.Linear(256, action_dim)  # 액션의 평균값
        self.fc_std = nn.Linear(256, action_dim)  # 액션의 표준편차
        self.fc_v = nn.Linear(256, action_dim)  # 상태 가치 함수
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def actor(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))  # 액션의 범위 [-2, 2]
        std = F.softplus(self.fc_std(x))  # 표준편차는 항상 양수여야 하므로 softplus 사용
        std = torch.clamp(std, min=1e-3)
        return mu, std

    def critic(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 10.0])  # 리워드 스케일링
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch = torch.tensor(s_lst, dtype=torch.float)
        a_batch = torch.tensor(a_lst, dtype=torch.float)
        r_batch = torch.tensor(r_lst, dtype=torch.float)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float)
        done_batch = torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.critic(s_prime) * done
        delta = td_target - self.critic(s)

        I = 1.0

        mu, std = self.actor(s)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(a)
        loss = -I * log_prob * delta.detach() + F.smooth_l1_loss(self.critic(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        I *= gamma

def main():
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = ActorCritic(state_dim, action_dim)
    print_interval = 20
    score = 0.0

    for n_epi in range(5000):
        done = False
        s, _ = env.reset()
        while not done:
            for t in range(n_rollout):
                mu, std = model.actor(torch.from_numpy(s).float())
                # print(mu, std)
                dist = Normal(mu, std)
                a = dist.sample()
                a = torch.clamp(a, -2.0, 2.0)  # 액션을 [-2, 2]로 클램핑
                s_prime, r, terminated, truncated, _  = env.step(np.array([a.item()], dtype=np.float32))
                done = terminated or truncated
                model.put_data((s, a.item(), r, s_prime, done))

                s = s_prime
                score += r

                if done:
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    # 최종 모델 저장
    torch.save(model.state_dict(), f"{model_dir}/actor_critic_final.pth")
    env.close()

if __name__ == '__main__':
    main()
