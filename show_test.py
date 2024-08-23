import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "Pendulum","score_log")

# 각 txt 파일의 경로를 리스트로 저장
files = os.listdir(model_dir)

# 각 모델에 대한 범례 이름
model_names = []
for name in files:
    i = name.split("test_")
    real_name = i[1].split(".txt")
    model_names.append(real_name[0])

# 각 모델의 결과를 저장할 리스트
all_results = []

# 모든 파일에서 결과를 읽어옴
for file_path in files:
    with open(model_dir + "/" + file_path, "r") as file:
        results = [float(line.strip()) for line in file]
        all_results.append(results)

# 그래프 그리기
plt.figure(figsize=(12, 6))

for idx, results in enumerate(all_results):
    plt.plot(results, label=model_names[idx], linewidth=2)  # 선 굵기 조절

plt.title("Comparison of Models", fontsize=16)
plt.xlabel("Test Episode", fontsize=14)
plt.ylabel("Total Reward", fontsize=14)
plt.ylim(-1600, 0)
plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)
plt.grid(False)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()
