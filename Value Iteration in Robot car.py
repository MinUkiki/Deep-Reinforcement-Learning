import numpy as np

# 상태와 행동 정의
states = ['cool', 'warm', 'overheated']
actions = ['slow', 'fast']

# 보상 정의
rewards = {
    'cool': {'slow': 1, 'fast': 2},
    'warm': {'slow': 1, 'fast': -10}
}

# 상태 전이 확률 정의
transition_probabilities = {
    'cool': {
        'slow': {'cool': 1.0},
        'fast': {'cool': 0.5, 'warm': 0.5}
    },
    'warm': {
        'slow': {'cool': 0.5, 'warm': 0.5},
        'fast': {'overheated': 1.0}
    },
}

# 파라미터 설정
gamma = 0.9  # 할인 인자
theta = 0.0001  # 수렴 기준

# 가치 함수 초기화
V = {state: 0 for state in states}

# Value Iteration 알고리즘
def value_iteration(states, actions, rewards, transition_probabilities, gamma, theta):
    V = {state: 0 for state in states}
    while True:
        delta = 0 
        for state in states:
            max_value = float('-inf') # 최댓값을 저장할 변수
            if state == 'overheated':  # 종료 상태의 경우 skip
                continue
            v = V[state]
            for action in actions:
                expected_value = 0  # 각 행동에 대한 기대 보상을 계산할 변수
                # 현재 상태에서 특정 행동을 취했을 때 가능한 모든 다음 상태에 대해 기대 보상을 합산
                for next_state, prob in transition_probabilities[state][action].items():
                    expected_value += prob * (rewards[state][action] + gamma * V[next_state])  # 기대 보상 계산

                # 최대 기대 보상을 선택
                if expected_value > max_value:
                    max_value = expected_value

            # 현재 상태의 가치 함수 값을 최대 기대 보상으로 갱신
            V[state] = max_value
            delta = max(delta, abs(v - V[state])) # 수렴되었는지 확인
        if delta < theta:
            break
    return V

# 최적 정책 추출
def policy(states, actions, rewards, transition_probabilities, V, gamma):
    policy = {}
    optimal_action = 0
    for state in states:
        p = 0
        optimal_p = float("-inf")
        if state == 'overheated':  # 종료 상태의 경우 정책은 None
            policy[state] = None
        else:
            for action in actions:
                for next_state, prob in transition_probabilities[state][action].items():
                    p += prob * (rewards[state][action] + gamma * V[next_state])
                if p > optimal_p:
                    optimal_p = p
                    optimal_action = action
            policy[state] = optimal_action
    return policy

# 가치 함수 계산
optimal_values = value_iteration(states, actions, rewards, transition_probabilities, gamma, theta)
print("Optimal Values:", optimal_values)

# 최적 정책 추출
optimal_policy = policy(states, actions, rewards, transition_probabilities, optimal_values, gamma)
print("Optimal Policy:", optimal_policy)
