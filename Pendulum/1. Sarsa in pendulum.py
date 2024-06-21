import numpy as np
import gymnasium as gym

env = gym.make('Pendulum-v1')

state = env.reset()

print(env)