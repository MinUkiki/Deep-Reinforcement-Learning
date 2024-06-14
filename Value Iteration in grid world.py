import numpy as np

# 그리드 월드 크기 정의
grid_height = 3
grid_width = 4

# 그리드 월드 초기화 (0: 빈 공간, -1: 장애물, 1: 시작 지점, 2: 목표 지점)
grid_world = np.zeros((grid_height, grid_width))

# 시작 지점과 목표 지점 설정
start_position = (grid_height-1, 0) # (row, col)
good_goal_position = (0, grid_width-1) # (row, col)
bad_goal_position = (1, grid_width-1) # (row, col)
grid_world[start_position] = 1
grid_world[good_goal_position] = 2
grid_world[bad_goal_position] = 2

# 장애물 위치 설정
obstacles = [(1, 1)]
for obstacle in obstacles:
    grid_world[obstacle] = -1

# 그리드 월드 출력
print(f"그리드 월드 : \n {grid_world} \n")

# action
actions = ['up', 'down', 'left', 'right']

# 주어진 상태와 행동에 대한 다음 상태와 전이 확률을 반환합니다.
def transition_probabilities(state, action):
    next_states = []
    prob = []
    if state == good_goal_position or state == bad_goal_position:
        # 목표 상태에 도달하면 다음 상태는 자기 자신이며 전이 확률은 1입니다.
        next_states.append(state)
        prob.append(1)
    else:
        for next_action in actions:
            if next_action == action:
                # 주어진 행동에 대한 다음 상태로 이동할 확률은 0.8입니다.
                next_states.append(get_next_state(state, next_action))
                prob.append(0.7)
            else:
                # 다른 방향으로 이동할 확률은 각각 0.1입니다.
                next_states.append(get_next_state(state, next_action))
                prob.append(0.1)
    return next_states, prob

# 보상
def get_reward(state):
    if good_goal_position == state:
        reward = 1
    elif bad_goal_position == state:
        reward = -1
    else:
        reward = 0
    return reward

# 에이전트의 현재 위치와 행동을 기반으로 새로운 위치를 반환
def get_next_state(state, action):
    """
    state: (row, col) 현재 위치
    action: str 이동 행동 ('up', 'down', 'left', 'right')
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
    if grid_world[next_state] == -1:
        # 장애물인 경우 원래 위치를 반환
        return state
    else:
        return next_state

def value_iteration(grid_height, grid_width, gamma, theta, max_iterations):
    # 초기 가치 함수 설정 (모든 상태에서의 가치는 0으로 초기화)
    V = np.zeros((grid_height, grid_width))
    # max_iterations 만큼 반복
    for _ in range(max_iterations):
        delta = 0
        # state ( 시작 지점이 (2,0)이기 때문 )
        for i in range(grid_height-1, -1, -1):
            for j in range(grid_width):
                max_value = float("-inf")
                state = (i,j)
                if state == obstacles[0]: # 장애물은 value를 계산 하지 않음
                    continue
                v = V[state]
                # 각 행동에 대한 기대 보상 계산
                for action in actions:
                    expected_value = 0
                    next_state, prob = transition_probabilities(state, action)
                    reward = get_reward(state)
                    for idx in range(len(next_state)):
                        expected_value += prob[idx] * (reward + gamma * V[next_state[idx]])
                    # 최대 기대 보상을 선택
                    if expected_value > max_value:
                        max_value = expected_value
                V[state] = max_value
                delta = max(delta, abs(v - V[state])) # 수렴되었는지 확인
        if delta < theta:
            break
    return V

def policy(grid_height, grid_width, V, gamma):
    optimal_policy_list = []
    optimal_action = 0
    for i in range(grid_height-1, -1, -1):
        policy_list = [] # 행 마다 최적 행동 저장
        for j in range(grid_width):
            state = (i,j)
            optimal_p = float("-inf")
            # 종료 상태 및 장애물의 경우 정책은 None
            if state == (1,1) or state == good_goal_position or state == bad_goal_position:  
                policy_list.append(None)
            else:
                for action in actions:
                    p = 0
                    next_state, prob = transition_probabilities(state, action)
                    reward = get_reward(state)
                    for idx in range(len(next_state)):
                        p +=  prob[idx] * (reward + gamma * V[next_state[idx]])
                    if p > optimal_p:
                        optimal_p = p
                        optimal_action = action
                policy_list.append(optimal_action)
        optimal_policy_list.append(policy_list)
    return optimal_policy_list
    
gamma=0.9
theta=0.0001
max_iterations=10

optimal_value = value_iteration(grid_height, grid_width, gamma, theta, max_iterations)
print(f"optimal value : \n {optimal_value} \n")

optimal_policy = policy(grid_height, grid_width, optimal_value, gamma)
optimal_policy[0], optimal_policy[2] = optimal_policy[2], optimal_policy[0]
print("optimal policy :")
for op_policy in optimal_policy:
    print(op_policy)