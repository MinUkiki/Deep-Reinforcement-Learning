# SAC test
import gymnasium as gym
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, "../saved_model")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc_mu = nn.Linear(128,action_dim)
        self.fc_std  = nn.Linear(128,action_dim)

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

# 테스트 환경 설정
env = gym.make('Pendulum-v1')

# 저장된 모델 로드
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy_net = Actor(state_dim, action_dim)
policy_net.load_state_dict(torch.load(f'{model_dir}/sac_actor_pendulum.pth'))
policy_net.eval()  # 평가 모드로 전환

num_test_episodes = 10
total_rewards = []

# 파일 열기 (쓰기 모드)
with open("test_sac.txt", "w") as file:
    for episode in range(num_test_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                mean, std = policy_net(state)
                action = mean # 평균값을 사용하여 액션을 결정

            next_state, reward, terminated, truncated, _ = env.step(np.array([action.item()], dtype=np.float32))
            done = terminated or truncated

            total_reward += reward
            state = torch.tensor(next_state, dtype=torch.float32)

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_test_episodes}, Total Reward: {total_reward}")

        # Total Reward 값을 파일에 저장
        file.write(f"{total_reward}\n")

# 평균 보상 계산 및 출력
average_reward = np.mean(total_rewards)
print(f"Average reward over {num_test_episodes} episodes: {average_reward}")
