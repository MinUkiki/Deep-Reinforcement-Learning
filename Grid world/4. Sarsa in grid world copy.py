import numpy as np
import random

height_size = 4
width_size = 4

class GridWorld:
    def __init__(self, height_size=height_size, width_size=width_size):
        self.height_size = height_size
        self.width_size = width_size
        self.reset()
        self.action_space = [0, 1, 2, 3] # 상 하 좌 우
        self.state_space = [(i, j) for i in range(height_size) for j in range(width_size)]
        self.goal_state = (height_size-1, width_size-1)
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        x, y = self.state
        if action == 0 and x > 0:  # 상
            x -= 1
        elif action == 1 and x < self.height_size - 1:  # 하
            x += 1
        elif action == 2 and y > 0:  # 좌
            y -= 1
        elif action == 3 and y < self.width_size - 1:  # 우
            y += 1
        self.state = (x, y)
        reward = 0 if self.state == self.goal_state else -1 # 보상 
        done = self.state == self.goal_state # 에피소드 종료 유무
        return self.state, reward, done, {}

    def render(self):
        grid = np.zeros((self.height_size, self.width_size), dtype=int)
        x, y = self.goal_state
        grid[x, y] = 1
        print(f"{grid}\n{'-'*50}")

env = GridWorld()
env.render()

class SarsaAgent:
    def __init__(self, gamma, alpha=0.1, epsilon=0.9):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.constant_epsilon = self.epsilon
        self.q_table = np.zeros((env.height_size,env.width_size,len(env.action_space)))
    
    def policy(self, state):
        x,y = state
        if random.random() < self.epsilon:
            return np.random.choice(env.action_space)
        else:
            return np.argmax(self.q_table[x,y])

    def train(self, episodes):
        for _ in range(episodes):
            done = False
            state = self.env.reset()
            action = self.policy(state)
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.policy(next_state)
                x,y = state
                next_x, next_y = next_state
                q_target = reward + self.gamma*self.q_table[next_x, next_y,next_action]
                self.q_table[x,y,action] += self.alpha*( q_target - self.q_table[x,y,action])
                
                state, action = next_state, next_action
    
# 에이전트 생성 및 학습
agent = SarsaAgent(gamma=0.99)
agent.train(episodes=1000)

# 행동 가치 함수 출력
for state in agent.env.state_space:
    print(f"State {state}: {agent.q_table[state]}")

def get_policy(Q):
    policy = np.zeros((height_size, width_size), dtype=int)
    for i in range(height_size):
        for j in range(width_size):
            policy[i, j] = np.argmax(Q[(i, j)])
    return policy

policy = get_policy(agent.q_table)
print(f"{'-'*50} \nOptimal Policy (0: Up, 1: Down, 2: Left, 3: Right): ")
print(policy)

def visualize_policy(policy):
    policy_symbols = np.array([['↑', '↓', '←', '→'][a] for a in policy.flatten()])
    policy_symbols = policy_symbols.reshape((height_size, width_size))
    for row in policy_symbols:
        print(' '.join(row))

print(f"{'-'*50} \nOptimal Policy Symbol:")
visualize_policy(policy)