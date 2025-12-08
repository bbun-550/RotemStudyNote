import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 학생 10명의 시험점수로 KMeans 수행 k=3
students = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10']
# 시험 점수
scores = np.array([76,95,65,85,60,92,55,88,83,72]).reshape(-1,1)
print(f'점수 : \n{scores.ravel()}')

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_clust = kmeans.fit_predict(scores)

df = pd.DataFrame({
    'Students':students,
    'Score':scores.ravel(),
    'Cluster':kmeans_clust
})
print(f'군집 결과 :\n{df}\n')

print(f'군집별 평균 점수')
grouped = df.groupby('Cluster')['Score'].mean()
print(grouped)

# 시각화
x_position = np.arange(len(students))
y_scores = scores.ravel()
colors = {0:'red',1:'blue', 2:'black'}
plt.figure(figsize=(10,6))
for i,(x,y,cluster) in enumerate(zip(x_position, y_scores, kmeans_clust)):
    plt.scatter(x,y,color=colors[cluster], s=100)
    # 점 찍고 글씨도 출력
    plt.text(x,y + 1.5, students[i], fontsize=10, ha='center')

# 중심점 표시
centers = kmeans.cluster_centers_
for center in centers:
    plt.scatter(len(students) // 2, center[0], marker='X', c='gold', s=200)


plt.xticks(x_position, students)
plt.xlabel('Students')
plt.ylabel('Score')
plt.title('KMeans Clustering of Student Scores')
plt.grid(True)
plt.show()
plt.close()
