import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from torch.distributions import Normal
from pendulum import PendulumEnv

# Hyperparameters
actor_learning_rate = 0.0002
critic_learning_rate = 0.001
gamma = 0.98
n_rollout = 10
target_update_interval = 10  # Target network 업데이트 주기
model_dir = "Pendulum\saved_model"

# 디렉토리가 없으면 생성
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc_mu = nn.Linear(256, action_dim)  # 액션의 평균값
        self.fc_std = nn.Linear(256, action_dim)  # 액션의 표준편차

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))  # 액션의 범위 [-2, 2]
        std = F.softplus(self.fc_std(x))  # 표준편차는 항상 양수여야 하므로 softplus 사용
        std = torch.clamp(std, min=1e-3)
        return mu, std

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc_v = nn.Linear(256, 1)  # 상태 가치 함수

    def forward(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.target_critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.update_target_network()
        self.data = []

    def update_target_network(self):
        """Target Critic 네트워크를 Critic 네트워크의 가중치로 업데이트"""
        self.target_critic.load_state_dict(self.critic.state_dict())

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([(r + 8) / 8])  # 리워드 스케일링
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

    def train_net(self, n_epi):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.target_critic(s_prime) * done  # Target Network를 사용하여 TD Target 계산
        critic_loss = F.smooth_l1_loss(self.critic(s), td_target.detach())

        # Critic 업데이트
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 업데이트
        mu, std = self.actor(s)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(a)
        actor_loss = -log_prob * (td_target - self.critic(s)).detach()

        self.actor_optimizer.zero_grad()
        actor_loss.mean().backward()
        self.actor_optimizer.step()

        # 일정 주기마다 Target Critic 네트워크 업데이트
        if n_epi % target_update_interval == 0:
            self.update_target_network()

def main():
    # env = gym.make('Pendulum-v1')
    env = PendulumEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = ActorCriticAgent(state_dim, action_dim)
    print_interval = 20
    score = 0.0

    for n_epi in range(1500):
        done = False
        s, _ = env.reset()
        while not done:
            for t in range(n_rollout):
                mu, std = agent.actor(torch.from_numpy(s).float())
                dist = Normal(mu, std)
                a = dist.sample()
                a = torch.clamp(a, -2.0, 2.0)  # 액션을 [-2, 2]로 클램핑
                s_prime, r, terminated, truncated, _ = env.step(np.array([a.item()], dtype=np.float32))
                done = terminated or truncated
                agent.put_data((s, a.item(), r, s_prime, done))

                s = s_prime
                score += r

                if done:
                    break

            agent.train_net(n_epi)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

        # 후반에 학습이 안되는 경우
        if n_epi % 100 == 0 and n_epi != 0:
            torch.save(agent.actor.state_dict(), f"{model_dir}/{n_epi}_actor_withTarget.pth")
            torch.save(agent.critic.state_dict(), f"{model_dir}/{n_epi}_critic_withTarget.pth")

    # 최종 모델 저장
    torch.save(agent.actor.state_dict(), f"{model_dir}/actor_withTarget_pendulum.pth")
    torch.save(agent.critic.state_dict(), f"{model_dir}/critic_withTarget_pendulum.pth")
    env.close()

if __name__ == '__main__':
    main()
