# DDPG test

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Actor 네트워크 정의 (MuNet)
class MuNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * 2  # Pendulum-v1의 액션 범위는 [-2, 2]
        return mu

# 테스트 환경 설정
env = gym.make('Pendulum-v1')

# 저장된 모델 로드
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Actor 네트워크 초기화 및 모델 로드
actor_net = MuNet(state_dim, action_dim)
actor_net.load_state_dict(torch.load('Pendulum\save_model\ddpg_actor_final.pth'))
actor_net.eval()  # 평가 모드로 전환

num_test_episodes = 10
total_rewards = []

# 파일 열기 (쓰기 모드)
with open("test_ddpg.txt", "w") as file:
    for episode in range(num_test_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                action = actor_net(state)  # 학습된 Actor 네트워크로부터 액션 결정

            # 액션을 환경에 전달하고 다음 상태, 보상, 종료 여부를 받음
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