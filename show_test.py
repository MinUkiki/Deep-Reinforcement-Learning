import matplotlib.pyplot as plt

# 각 txt 파일의 경로를 리스트로 저장
file_paths = [
    "results_model_1.txt",  # 모델 1의 결과 파일
    # 추가적인 결과 파일 경로를 여기에 추가
]

# 각 모델에 대한 범례 이름
model_names = [
    "Baseline Model",       # 모델 1의 범례 이름
    # 추가적인 범례 이름을 여기에 추가
]

# 각 모델의 결과를 저장할 리스트
all_results = []

# 모든 파일에서 결과를 읽어옴
for file_path in file_paths:
    with open(file_path, "r") as file:
        results = [float(line.strip()) for line in file]
        all_results.append(results)

# 그래프 그리기
plt.figure(figsize=(10, 6))

for idx, results in enumerate(all_results):
    plt.plot(results, marker='o', label=model_names[idx])  # 범례 이름을 직접 지정

plt.title("Comparison of Models")
plt.xlabel("Test Episode")
plt.ylabel("Total Reward")
plt.legend()  # 범례 추가
plt.grid(True)
plt.show()
