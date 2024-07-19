import numpy as np
import matplotlib.pyplot as plt

# 정규 분포의 매개변수
mu = 0
sigma = 1

# 확률 밀도 함수 계산
def normal_pdf(x, mu, sigma):
    coefficient = 1 / (np.sqrt(2 * np.pi * sigma ** 2))
    exponential = np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
    return coefficient * exponential

# 특정 값에서의 확률 밀도
x_val = 1
pdf_x_val = normal_pdf(x_val, mu, sigma)
print(f"PDF at x = {x_val}: {pdf_x_val}")

# 사다리꼴 적분법을 사용한 수치 적분 함수
def trapezoidal_integral(f, a, b, n, *args):
    x = np.linspace(a, b, n)
    y = f(x, *args)
    h = (b - a) / (n - 1)
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral

# 특정 구간에서의 확률 계산
a, b = 0.5, 1.5
n = 10000  # 적분 구간의 개수
prob = trapezoidal_integral(normal_pdf, a, b, n, mu, sigma)
print(f"P({a} <= X <= {b}): {prob}")

# 그래프 그리기
x = np.linspace(-3, 3, 1000)
pdf = normal_pdf(x, mu, sigma)

x_range = np.linspace(a, b, 1000)
pdf_range = normal_pdf(x_range, mu, sigma)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, label='PDF of N(0, 1)', color='blue')
plt.fill_between(x_range, pdf_range, alpha=0.5, label=f'P({a} <= X <= {b})', color='green')
plt.scatter(x_val, pdf_x_val, color='red', zorder=5)
plt.vlines(x_val, ymin=0, ymax=pdf_x_val, colors='red', linestyles='dotted', label=f'PDF at x = {x_val}')
plt.title('Normal Distribution PDF')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
