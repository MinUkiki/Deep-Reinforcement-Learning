import numpy as np
from collections import defaultdict
'''
다양한 MC 방법들
'''
first_visit = True # first or every
constant_alpha = False # Constant-alpha
use_epsilon = True
size = 4

class GridWorld:
    def __init__(self, size=size):
        self.size = size
        self.reset()
        self.action_space = [0, 1, 2, 3] # 상 하 좌 우
        self.state_space = [(i, j) for i in range(size) for j in range(size)]
        self.goal_state = (size-1, size-1)
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        x, y = self.state
        if action == 0 and x > 0:  # 상
            x -= 1
        elif action == 1 and x < self.size - 1:  # 하
            x += 1
        elif action == 2 and y > 0:  # 좌
            y -= 1
        elif action == 3 and y < self.size - 1:  # 우
            y += 1
        self.state = (x, y)
        reward = 1 if self.state == self.goal_state else 0 # 보상 
        done = self.state == self.goal_state # 에피소드 종료 유무
        return self.state, reward, done, {}

    def render(self):
        grid = np.zeros((self.size, self.size), dtype=int)
        x, y = self.state
        grid[x, y] = 1
        print(grid)

env = GridWorld()

class MonteCarloAgent:
    def __init__(self, gamma=1.0, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_table = defaultdict(float) # 상태 가치 함수
        self.returns = defaultdict(list)
    
    # PE
    def generate_episode(self, policy):
        episode = []
        done = False
        state = self.env.reset()
        while not done:
            action = policy(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode
    
    def update_value_function(self, episode, alpha):
        visited_states = set()
        G = 0
        for state, _, reward in reversed(episode):
            if first_visit:
                if state not in visited_states:
                    visited_states.add(state)
                else:
                    continue
            G = self.gamma * G + reward
            self.returns[state].append(G)
            if constant_alpha:
                self.value_table[state] = (1-alpha)*self.value_table[state] + alpha*G
            else:
                self.value_table[state] = np.mean(self.returns[state])
        
    def policy(self, state):
        # if use_epsilon: # epsilon 사용 여부
        #     action = np.argmax(self.value_table[state])
        #     a = 1-self.epsilon + self.epsilon/len(self.env.action_space)
        #     if np.random.rand() < a:
        #         return action
        #     else:
        #         pass
        # else:
        #     return np.random.choice(env.action_space)
        return np.random.choice(env.action_space)
    
    def train(self, episodes):
        alpha = 0.001
        for _ in range(episodes):
            episode = self.generate_episode(self.policy)
            self.update_value_function(episode,alpha)

# 에이전트 생성 및 학습
agent = MonteCarloAgent(gamma=0.9)
agent.train(episodes=5000)

table = []
for i in range(size):
    row = [0] * size  # 각 행을 [0, 0, 0, 0]으로 초기화
    table.append(row)

# 상태 가치 함수 출력
for state, value in agent.value_table.items():
    table[state[0]][state[1]] = value

for a in table:
    print(', '.join(map(str, a)))