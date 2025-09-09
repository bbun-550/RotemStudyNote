# 가상의 데이터로 쇼핑몰 고객 세분화(집단화)
# DBSCAN 군집화 - 표준화 추천
# - 데이터 : 고객 수, 연간지출액, 방문 수 ...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

np.random.seed(0)
n_customers = 200 # 고객수
annual_spending = np.random.normal(50000, 15000, n_customers) # 연간 지출액 
monthly_visit = np.random.normal(5, 2, n_customers) # 월 방문 횟수
# print(annual_spending, monthly_visit)

# np.clip() 수치 안정화, 범위 고정 등에 사용
annual_spending = np.clip(annual_spending, 0, None)
monthly_visit = np.clip(monthly_visit, 0, None)
# print(annual_spending[:5], monthly_visit[:5])
'''
[76460.78518951 56002.35812551 64681.06976159 83613.39798802 78013.36985225] 
[4.26163632 4.52124164 7.19931919 6.31052746 6.28026305]
'''

# data를 DataFrame에 넣기
data = pd.DataFrame({
    'annual_spending': annual_spending, 'monthly_visit': monthly_visit})
# print(data.head())
'''
   annual_spending  monthly_visit
0     76460.785190       4.261636
1     56002.358126       4.521242
2     64681.069762       7.199319
3     83613.397988       6.310527
4     78013.369852       6.280263
'''

# 표준화
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
# print(data_scaled[:2])
'''
[[ 1.65763081 -0.25953531]
 [ 0.32234127 -0.12108587]]
'''

dbscan = DBSCAN(eps=0.35, min_samples=4)
# - eps : 반경을 의미한다. 너무 작아지면 노이즈가 많아지고, 커지면 모두 한 군집에 속하게 된다.
clusters = dbscan.fit_predict(data_scaled)
data['cluster'] = clusters
print(data.head())
'''
   annual_spending  monthly_visit  cluster
0     76460.785190       4.261636       -1
1     56002.358126       4.521242        0
2     64681.069762       7.199319        0
3     83613.397988       6.310527       -1
4     78013.369852       6.280263       -1
'''

# 시각화
for cluster_id in np.unique(clusters):
    cluster_data = data[data['cluster'] == cluster_id]
    plt.scatter(cluster_data['annual_spending'], cluster_data['monthly_visit'], label=f'cluster{cluster_id}', s=10)
# plt.scatter(dbscan.cluster_centers_[:,0],dbscan.cluster_centers_[:,1], marker='X', s=200, c='r', label='centroid')
plt.xlabel('annual_spending')
plt.ylabel('monthly_visit')
plt.legend()
plt.show()
plt.close()

print(data['cluster'].value_counts())
'''
 0    126
-1     64 # 이상치/노이즈
 1     10
'''