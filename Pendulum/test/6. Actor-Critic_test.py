# Actor Critic test
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(state_dim, 256)  # Pendulum의 상태 공간 크기는 3
        self.fc_mu = nn.Linear(256, action_dim)  # 액션의 평균값
        self.fc_std = nn.Linear(256, action_dim)  # 액션의 표준편차
        self.fc_v = nn.Linear(256, action_dim)  # 상태 가치 함수
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)

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

# 테스트 환경 설정
env = gym.make('Pendulum-v1')

# 저장된 모델 로드
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy_net = ActorCritic(state_dim, action_dim)
policy_net.load_state_dict(torch.load('Pendulum\save_model\sac_actor_final.pth'))
policy_net.eval()  # 평가 모드로 전환

num_test_episodes = 10
total_rewards = []

# 파일 열기 (쓰기 모드)
with open("test_actor_critic.txt", "w") as file:
    for episode in range(num_test_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                mean, std = policy_net.actor(state)
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