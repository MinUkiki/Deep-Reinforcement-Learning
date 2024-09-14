import gymnasium as gym
import torch, os, sys
import numpy as np
import torch.nn as nn

env_dir= os.path.dirname(os.path.abspath(__file__))
pendulm_dir = os.path.dirname(env_dir)
sys.path.append(pendulm_dir)
from pendulum import PendulumEnv

current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, "../saved_model")

# Q-네트워크 정의
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)
    
# env = gym.make('Pendulum-v1')
env = PendulumEnv(render_mode='human')

# 네트워크 초기화
state_dim = env.observation_space.shape[0]
action_dim = 11
q_network = QNetwork(state_dim, action_dim)

# 모델 불러오기
q_network.load_state_dict(torch.load(f'{model_dir}\double_dqn_pendulum.pth')) # Double DQN
q_network.eval()

# 이산적 액션 공간을 연속적 액션 공간으로 변환하는 함수
def discrete_to_continuous(action_idx, action_space):
    """이산적인 액션 인덱스를 연속적 액션 값으로 변환"""
    return np.array([action_space[action_idx]], dtype=np.float32)

# 이산적 액션 공간 설정 (예: [-2, 2]를 11개의 구간으로 나누기)
action_space = np.linspace(-2.0, 2.0, action_dim)

# 테스트 에피소드 수 설정
num_test_episodes = 10
total_rewards = []

# 파일 열기 (쓰기 모드)
with open("test_double_dqn.txt", "w") as file:
    for episode in range(num_test_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor([state], dtype=torch.float32)
            with torch.no_grad():
                action_idx = q_network(state_tensor).argmax().item()
                action = discrete_to_continuous(action_idx, action_space)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_test_episodes}, Total Reward: {total_reward}")
        
        # Total Reward 값을 파일에 저장
        file.write(f"{total_reward}\n")

# 평균 보상 계산 및 출력
average_reward = np.mean(total_rewards)
print(f"Average reward over {num_test_episodes} episodes: {average_reward}")