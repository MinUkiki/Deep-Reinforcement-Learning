# import gymnasium as gym # 다른 환경을 사용할 때
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import os
import numpy as np
from pendulum import PendulumEnv

# 하이퍼파라미터
learning_rate = 0.0003
gamma = 0.9
lmbda = 0.9
eps_clip = 0.2
K_epoch = 10
rollout_len = 3
buffer_size = 10
minibatch_size = 128

# 경로 설정
current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, "saved_model")

# 디렉토리가 없으면 생성
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 액터 네트워크 정의
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc_mu = nn.Linear(128, action_dim)
        self.fc_std = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

# 크리틱 네트워크 정의
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc_v = nn.Linear(128, 1)  # 크리틱은 상태 가치 V(s)를 출력합니다.

    def forward(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

# PPO 알고리즘 클래스 정의
class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.data = []
        self.optimization_step = 0

    # 데이터를 버퍼에 추가하는 함수
    def put_data(self, transition):
        self.data.append(transition)
    
    # 미니배치를 만드는 함수
    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(buffer_size):
            for i in range(minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition
                    
                    s_lst.append(s)
                    a_lst.append([a])
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)
                    
            mini_batch = torch.tensor(s_batch, dtype=torch.float), torch.tensor(a_batch, dtype=torch.float), \
                          torch.tensor(r_batch, dtype=torch.float), torch.tensor(s_prime_batch, dtype=torch.float), \
                          torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)
            data.append(mini_batch)

        return data

    # 어드밴티지를 계산하는 함수
    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + gamma * self.critic(s_prime) * done_mask
                delta = td_target - self.critic(s)
            delta = delta.numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

    # 네트워크를 학습하는 함수
    def train_net(self):
        if len(self.data) == minibatch_size * buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self.actor(s)
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
                    actor_loss = -torch.min(surr1, surr2)
                    critic_loss = F.smooth_l1_loss(self.critic(s), td_target)

                    self.optimizer_actor.zero_grad()
                    self.optimizer_critic.zero_grad()
                    actor_loss.mean().backward()
                    critic_loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.optimizer_actor.step()
                    self.optimizer_critic.step()
                    self.optimization_step += 1
        
# 메인 함수 정의
def main():
    # env = gym.make('Pendulum-v1')
    env = PendulumEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPO(state_dim, action_dim)
    score = 0.0
    print_interval = 20
    rollout = []

    for n_epi in range(1500):
        s, _ = env.reset()
        done = False
        count = 0
        while count < 200 and not done:
            for t in range(rollout_len):
                mu, std = agent.actor(torch.from_numpy(s).float())
                dist = Normal(mu, std)
                a = dist.sample()
                log_prob = dist.log_prob(a)
                s_prime, r, terminated, truncated, _ = env.step(np.array([a.item()], dtype=np.float32))
                done = terminated or truncated

                rollout.append((s, a, r/10.0, s_prime, log_prob.item(), done))
                if len(rollout) == rollout_len:
                    agent.put_data(rollout)
                    rollout = []

                s = s_prime
                score += r
                count += 1

            agent.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"# of episode :{n_epi}, avg score : {score / print_interval:.1f}")
            score = 0.0

        if n_epi % 100 == 0 and n_epi != 0:
            torch.save(agent.actor.state_dict(), f"{model_dir}/{n_epi}_ppo_actor.pth")
            torch.save(agent.critic.state_dict(), f"{model_dir}/{n_epi}_ppo_critic.pth")

    # 최종 모델 저장
    torch.save(agent.actor.state_dict(), f"{model_dir}/ppo_actor_pendulum.pth")
    torch.save(agent.critic.state_dict(), f"{model_dir}/ppo_critic_pendulum.pth")
    env.close()

if __name__ == '__main__':
    main()
