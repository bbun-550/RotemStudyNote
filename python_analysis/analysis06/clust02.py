# 계층적 군집
# 10명 학생의 시험점수를 사용

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

students = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10']
scores = np.array([76,95,65,85,60,92,55,88,83,72]).reshape(-1,1)
print(f'점수 : \n{scores.ravel()}') # flatten 보단 ravel를 사용해라(이유 : 더 빠르고, 복사를 하지 않음)

# 계층적 군집
linked = linkage(scores, method='ward')

plt.figure(figsize=(10,6))
dendrogram(linked, labels=students)
plt.axhline(y=25, color='r', linestyle='--', label='cut at height=25') # horizontal line
plt.xlabel('Students')
plt.ylabel('Distance')
plt.legend()
plt.grid(True)
plt.show()
plt.close()

# 군집 3개로 나누기
clusters = fcluster(linked, 3, criterion='maxclust')
print(clusters) # [2 1 3 1 3 1 3 1 1 2]

for student, cluster in zip(students, clusters):
    print(f'{student} : cluster {cluster}')
'''
s1 : cluster 2
s2 : cluster 1
s3 : cluster 3
s4 : cluster 1
s5 : cluster 3
s6 : cluster 1
s7 : cluster 3
s8 : cluster 1
s9 : cluster 1
s10 : cluster 2
'''

# 군집별로 점수와 이름 정리
cluster_info = {}
for student, cluster, score in zip(students, clusters, scores.ravel()):
    if cluster not in cluster_info:
        cluster_info[cluster] = {'students':[], 'scores':[]}
    cluster_info[cluster]['students'].append(student)
    cluster_info[cluster]['scores'].append(score)

for cluster_id, info in sorted(cluster_info.items()):
    avg_score = np.mean(info['scores'])
    student_list = ', '.join(info['students'])
    print(f'Cluster {cluster_id} : 평균점수={avg_score:.2f}, 학생들={student_list}')
  

# 군집 시각화
x_position = np.arange(len(students))
y_scores = scores.ravel()
colors={1:'red', 2:'blue', 3:'green'}
plt.figure(figsize=(10,6))
for i,(x,y,cluster) in enumerate(zip(x_position, y_scores, clusters)):
    plt.scatter(x,y,color=colors[cluster], s=100)
    # 점 찍고 글씨도 출력
    plt.text(x,y + 1.5, students[i], fontsize=10, ha='center')

plt.xticks(x_position, students)
plt.xlabel('Students')
plt.ylabel('Score')
plt.grid(True)
plt.show()
plt.close()
