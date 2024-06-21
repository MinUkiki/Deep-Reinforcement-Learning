import numpy as np
import gymnasium as gym

import numpy as np

# Q-table을 사용하기 위해 연속적인 state, action을 이산화 해야함

num_bins = 12
angle_bins = np.linspace(-1.0, 1.0, num_bins + 1)[1:-1]
velocity_bins = np.linspace(-8.0, 8.0, num_bins + 1)[1:-1]

env = gym.make('Pendulum-v1')

# def policy(self, state):
#     x,y = state
#     if np.random.rand() < self.epsilon:
#         return np.random.choice(env.action_space)
#     else:
#         return np.argmax(self.q_table[x,y])

state = env.reset()
angle, velocity = state
action = env.action_space.sample()

next_state, reward, terminated, truncated, info = env.step(action)

# env.action_space.sample()

print(env)