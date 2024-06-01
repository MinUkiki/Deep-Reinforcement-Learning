import numpy as np
# 그리드 월드 크기 정의
grid_height = 4
grid_width = 4
# 그리드 월드 초기화 (0: 빈 공간, 2: 목표 지점)
grid_world = np.zeros((grid_height, grid_width))

# 시작 지점과 목표 지점 설정 (row, col)
goal_position = [(0, 0), (grid_height-1,grid_width-1)] 

grid_world[goal_position[0]] = 2
grid_world[goal_position[1]] = 2

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

def policy_evaluation(gamma, theta, max_iteration):
    V = np.zeros((grid_height, grid_width))
    for _ in range(max_iteration):
        for i in range(grid_height):
            for j in range(grid_width):
                state = (i,j)
                delta = 0
                if state == goal_position[0] or state == goal_position[1]:
                    continue
                v = V[state]
                for action in actions:
                    prob = 0.25
                    next_state = get_next_state(state, action)
                    expected_value += prob * (reward + V[next_state])
                delta = max(delta, expected_value)
        if delta > theta:
            break

def policy_improvement():
    pass
