# 가상의 데이터로 쇼핑몰 고객 세분화(집단화)
# KMeans 군집화
# - 데이터 : 고객 수, 연간지출액, 방문 수 ...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

np.random.seed(0)
n_customers = 200 # 고객수
annual_spending = np.random.normal(50000, 15000, n_customers) # 연간 지출액 
monthly_visit = np.random.normal(5, 2, n_customers) # 월 방문 횟수
# print(annual_spending, monthly_visit)

# np.clip() 수치 안정화, 범위 고정 등에 사용
annual_spending = np.clip(annual_spending, 0, None)
monthly_visit = np.clip(monthly_visit, 0, None)
# print(annual_spending[:5], monthly_visit[:5])

# data를 DataFrame에 넣기
data = pd.DataFrame({
    'annual_spending': annual_spending, 'monthly_visit': monthly_visit})
print(data.head())

# 시각화 - 산포도
# plt.scatter(data['annual_spending'], data['monthly_visit'])
# plt.xlabel('annual_spending')
# plt.ylabel('monthly_visit')
# plt.show()
# plt.close()

# KMeans 군집화
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(data)

# 군집결과 시각화
data['clusters'] = clusters
print(data.head(5))

for cluster_id in np.unique(clusters):
    cluster_data = data[data['clusters'] == cluster_id]
    plt.scatter(cluster_data['annual_spending'], cluster_data['monthly_visit'], label=f'cluster{cluster_id}', s=10)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], marker='X', s=200, c='r', label='centroid')
plt.xlabel('annual_spending')
plt.ylabel('monthly_visit')
plt.legend()
plt.show()
plt.close()
