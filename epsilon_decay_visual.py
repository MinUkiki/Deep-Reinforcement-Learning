import matplotlib.pyplot as plt

# 초기 값
epsilon_initial = 1.0
steps = 100

# 감소율
decay_rate_1 = 0.998
decay_rate_2 = 0.98

# 에이전트 업데이트 시나리오
epsilon_1 = epsilon_initial
epsilon_2 = epsilon_initial

epsilon_values_1 = [epsilon_1]
epsilon_values_2 = [epsilon_2]

for step in range(steps):
    epsilon_1 *= decay_rate_1
    epsilon_2 *= decay_rate_2
    epsilon_values_1.append(epsilon_1)
    epsilon_values_2.append(epsilon_2)

# 시각화
plt.plot(epsilon_values_1, label='Decay Rate 0.998')
plt.plot(epsilon_values_2, label='Decay Rate 0.98')
plt.xlabel('Steps')
plt.ylabel('Epsilon')
plt.legend()
plt.title('Epsilon Decay Comparison')
plt.show()
