import numpy as np

# 그리드 월드 크기 정의
grid_height = 4
grid_width = 4

# 그리드 월드 초기화 (0: 빈 공간, 2: 목표 지점)
grid_world = np.zeros((grid_height, grid_width))

# 시작 지점과 목표 지점 설정 (row, col)
goal_positions = [(0, 0), (grid_height - 1, grid_width - 1)]

grid_world[goal_positions[0]] = 2
grid_world[goal_positions[1]] = 2

# 그리드 월드 출력
print(f"그리드 월드 : \n {grid_world} \n")

actions = ['up', 'down', 'left', 'right']
reward = -1

# 에이전트의 현재 위치와 행동을 기반으로 새로운 위치를 반환
def get_next_state(state, action):
    """
    state: (row, col) 현재 위치
    action: str 행동 ('up', 'down', 'left', 'right')
    """
    row, col = state
    if action == 'up' and row > 0:
        row -= 1
    elif action == 'down' and row < grid_height - 1:
        row += 1
    elif action == 'left' and col > 0:
        col -= 1
    elif action == 'right' and col < grid_width - 1:
        col += 1

    next_state = (row, col)
    return next_state

def policy_evaluation(policy, gamma, theta, max_iterations):
    V = np.zeros((grid_height, grid_width))
    for _ in range(max_iterations):
        delta = 0
        for i in range(grid_height):
            for j in range(grid_width):
                state = (i, j)
                if state in goal_positions:
                    continue
                v = V[state]
                expected_value = 0
                for action, action_prob in policy[state].items():
                    next_state = get_next_state(state, action)
                    expected_value += action_prob * (reward + gamma * V[next_state])
                V[state] = expected_value
                delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

def policy_improvement(V, gamma):
    new_policy = {}
    for i in range(grid_height):
        for j in range(grid_width):
            state = (i, j)
            new_policy[state] = {}
            if state in goal_positions:
                for a in actions:
                    new_policy[state][a] = 1.0
            else:
                action_values = {}
                for action in actions:
                    next_state = get_next_state(state, action)
                    action_values[action] = reward + gamma * V[next_state]
                
                best_action = max(action_values, key=action_values.get)
                for a in actions:
                    if a == best_action:
                        new_policy[state][a] = 1.0
                    else:
                        new_policy[state][a] = 0.0
    return new_policy

def policy_iteration(gamma=0.9, epsilon=1e-4, max_iterations=1000):
    policy = {}
    for i in range(grid_height):
        for j in range(grid_width):
            # 초기 정책: 모든 행동에 동일 확률
            policy[(i, j)] = {a: 0.25 for a in actions}  

    stable_policy = False
    while not stable_policy:
        V = policy_evaluation(policy, gamma, epsilon, max_iterations)
        new_policy = policy_improvement(V, gamma)

        stable_policy = True
        for state in policy:
            if policy[state] != new_policy[state]:
                stable_policy = False
                break
        policy = new_policy

    return V

optimal_value_function = policy_iteration()

print("\nOptimal Value Function:")
print(optimal_value_function)
