import numpy as np
import gymnasium as gym # 다른 환경을 사용할 때
import os
from pendulum import PendulumEnv

render = False

# 경로 설정
current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, "saved_model")
file_path = 'Qlearning_Q_table.npy'

if render:
    env = PendulumEnv(render_mode="human")
else:
    env = PendulumEnv()

# Q-table을 사용하기 위해 연속적인 state, action을 이산화

angle_num = 17
velocity_num = 17
action_num = 11

angle_bins = [-3.14, -2.74, -2.35, -1.96, -1.57,
            -1.17, -0.78, -0.39,  0., 0.39, 0.78,
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

def policy(angle, velocity):
    if np.random.rand() < epsilon:
        action = np.random.choice(range(action_num))
    else:
        action = np.argmax(q_table[angle, velocity])
    return action_bins[action]

episodes = 50000
gamma = 0.99
alpha = 0.1
epsilon = 0.1
q_table = np.zeros((angle_num, velocity_num, action_num))

# 학습 결과 추적을 위한 변수
total_rewards = []

for episode in range(episodes):
    terminated = False
    truncated = False
    state, _ = env.reset()
    angle, velocity = discrete_state(state)
    episode_reward = 0

    while not terminated and not truncated:
        action = policy(angle, velocity)
        next_state, reward, terminated, truncated, _ = env.step([action])
        next_angle, next_velocity = discrete_state(next_state)

        dis_action = discrete_action(action)

        q_target = reward + gamma * np.max(q_table[next_angle, next_velocity])
        q_table[angle, velocity, dis_action] += alpha * (q_target - q_table[angle, velocity, dis_action])

        angle, velocity = next_angle, next_velocity
        episode_reward += reward

    total_rewards.append(episode_reward)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}, Total Reward: {episode_reward}")

# 전체 학습 결과 출력
print("Training completed")
print(f"Average reward over last 100 episodes: {np.mean(total_rewards[-100:])}")

# Q-table 저장
np.save(model_dir + "/" + file_path, q_table)

env.close
