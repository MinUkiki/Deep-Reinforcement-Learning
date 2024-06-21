import numpy as np
import gymnasium as gym

env = gym.make('Pendulum-v1')

# Q-table을 사용하기 위해 연속적인 state, action을 이산화
angle_num_bins = 17
velocity_num_bins = 17
action_num_bin = 11
angle_bins = np.linspace(-np.pi, np.pi, angle_num_bins)
velocity_bins = np.linspace(-8.0, 8.0, velocity_num_bins)
action_bins = np.linspace(-2.0, 2.0, action_num_bin)

def discrete_state(state):
    radian = np.arctan2(state[1], state[0])
    radian_idx = np.digitize(radian, angle_bins) - 1
    velocity_idx = np.digitize(state[2], velocity_bins) - 1
    return radian_idx, velocity_idx

def discrete_action(action):
    return np.digitize(action, action_bins) - 1

def policy(angle, velocity):
    if np.random.rand() < epsilon:
        action = np.random.choice(range(action_num_bin))
    else:
        action = np.argmax(q_table[angle, velocity])
    return np.array([action_bins[action]], dtype=np.float32)

episodes = 100
gamma = 0.99
alpha = 0.1
epsilon = 0.9
q_table = np.zeros((angle_num_bins, velocity_num_bins, action_num_bin))

# 학습 결과 추적을 위한 변수
total_rewards = []

for episode in range(episodes):
    terminated = False
    truncated = False
    state, _ = env.reset()
    angle, velocity = discrete_state(state)
    action = policy(angle, velocity)
    episode_reward = 0

    while not terminated and not truncated:
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_angle, next_velocity = discrete_state(next_state)
        next_action = policy(next_angle, next_velocity)

        dis_action = discrete_action(action)
        dis_next_action = discrete_action(next_action)

        q_target = reward + gamma * q_table[next_angle, next_velocity, dis_next_action[0]]
        q_table[angle, velocity, dis_action[0]] += alpha * (q_target - q_table[angle, velocity, dis_action[0]])

        angle, velocity, action = next_angle, next_velocity, next_action
        episode_reward += reward

    total_rewards.append(episode_reward)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}, Total Reward: {episode_reward}")

# 전체 학습 결과 출력
print("Training completed")
print(f"Average reward over last 100 episodes: {np.mean(total_rewards[-100:])}")
