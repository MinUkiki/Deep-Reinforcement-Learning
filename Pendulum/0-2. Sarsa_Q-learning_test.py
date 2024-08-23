import numpy as np
import gymnasium as gym # 다른 환경을 사용할 때
from pendulum import PendulumEnv

render = True  # 테스트 시에는 렌더링을 켜서 시각적으로 확인
# file_path = 'Pendulum/save_model/Saras_Q_table.npy'
file_path = 'Pendulum/save_model/Qlearning_Q_table.npy'

if render:
    env = PendulumEnv(render_mode="human")
else:
    env = PendulumEnv()

# Q-table을 사용하기 위해 연속적인 state, action을 이산화
angle_num = 17
velocity_num = 17
action_num = 11

angle_bins = [-3.14, -2.74, -2.35, -1.96, -1.57,
            -1.17, -0.78, -0.39,  0.,  0.39, 0.78,
            1.17,  1.57,  1.96,  2.35, 2.74,  3.14]
velocity_bins = [-8., -7., -6., -5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4., 5.,  6.,  7.,  8.]
action_bins = [-2. , -1.6, -1.2, -0.8, -0.4,  0. ,  0.4,  0.8,  1.2,  1.6,  2. ]

def discrete_state(state):
    radian = np.arctan2(state[1], state[0])
    radian_idx = np.digitize(radian, angle_bins) - 1
    velocity_idx = np.digitize(state[2], velocity_bins) - 1
    return radian_idx, velocity_idx

def discrete_action(action):
    return np.digitize(np.array(action), action_bins) - 1

def policy(angle, velocity, q_table):
    action = np.argmax(q_table[angle, velocity])
    return action_bins[action]

# 학습된 Q-테이블 로드
q_table = np.load(file_path)

episodes = 10  # 테스트 에피소드 수
total_rewards = []

for episode in range(episodes):
    terminated = False
    truncated = False
    state, _ = env.reset()
    angle, velocity = discrete_state(state)
    episode_reward = 0

    while not terminated and not truncated:
        action = policy(angle, velocity, q_table)
        next_state, reward, terminated, truncated, _ = env.step([action])
        next_angle, next_velocity = discrete_state(next_state)

        angle, velocity = next_angle, next_velocity
        episode_reward += reward

    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1}, Total Reward: {episode_reward}")

# 전체 테스트 결과 출력
print("Testing completed")
print(f"Average reward over {episodes} episodes: {np.mean(total_rewards)}")

env.close