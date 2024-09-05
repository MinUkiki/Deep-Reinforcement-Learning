# SAC
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random, os
from pendulum import PendulumEnv

# Hyperparameters
actor_learning_rate = 0.0003
critic_learning_rate = 0.001
alpha_learning_rate = 0.001
gamma = 0.99
tau = 0.01
target_entropy = -1.0
init_alpha = 0.01
batch_size = 32
buffer_limit = 50000
current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, "saved_model")

# 디렉토리가 없으면 생성
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Replay Buffer
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

# Actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc_mu = nn.Linear(128,action_dim)
        self.fc_std  = nn.Linear(128,action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=actor_learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action) * 2.0
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s,a), q2(s,a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

# Critic
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc_s = nn.Linear(state_dim, 64)
        self.fc_a = nn.Linear(action_dim,64)
        self.fc_cat = nn.Linear(128,32)
        self.fc_out = nn.Linear(32,1)
        self.optimizer = optim.Adam(self.parameters(), lr=critic_learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a) , target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

# Calculate TD Target
def calc_target(actor, q1, q2, mini_batch):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob= actor(s_prime)
        entropy = -actor.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target

# Main Training Loop
def main():
    env = PendulumEnv()
    memory = ReplayBuffer()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    q1, q2, = Critic(state_dim, action_dim), Critic(state_dim, action_dim)
    q1_target, q2_target = Critic(state_dim, action_dim), Critic(state_dim, action_dim)
    actor = Actor(state_dim, action_dim)

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    score = 0.0
    print_interval = 20

    for n_epi in range(1500):
        s, _ = env.reset()
        done = False
        count = 0

        while count < 200 and not done:
            a, log_prob= actor(torch.from_numpy(s).float())
            s_prime, r, terminated, truncated, _ =  env.step(np.array([2.0*a.item()], dtype=np.float32))
            done = terminated or truncated
            memory.put((s, a.item(), r/10.0, s_prime, done))
            score +=r
            s = s_prime
            count += 1

        if memory.size()>1000:
            for i in range(20):
                mini_batch = memory.sample(batch_size)
                td_target = calc_target(actor, q1_target, q2_target, mini_batch)
                q1.train_net(td_target, mini_batch)
                q2.train_net(td_target, mini_batch)
                entropy = actor.train_net(q1, q2, mini_batch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)

        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"# of episode :{n_epi}, avg score : {score / print_interval:.1f}")
            score = 0.0

        if n_epi % 100 == 0 and n_epi != 0:
            torch.save(actor.state_dict(), f"{model_dir}/{n_epi}_sac_actor.pth")
            torch.save(q1.state_dict(), f"{model_dir}/{n_epi}_sac_critic1.pth")
            torch.save(q2.state_dict(), f"{model_dir}/{n_epi}_sac_critic2.pth")

    # 최종 모델 저장
    torch.save(actor.state_dict(), f"{model_dir}/sac_actor_pendulum.pth")
    torch.save(q1.state_dict(), f"{model_dir}/sac_critic1_pendulum.pth")
    torch.save(q2.state_dict(), f"{model_dir}/sac_critic2_pendulum.pth")
    env.close()

if __name__ == '__main__':
    main()
