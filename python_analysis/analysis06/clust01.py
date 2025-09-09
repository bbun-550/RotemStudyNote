# 클러스터링 기법 중 계층적 군집화 이해
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='applegothic')

np.random.seed(123)
var = ['X','Y'] # 좌표평면 올리기 위함
labels = ['점0','점1','점2','점3','점4']
X = np.random.random_sample([5,2]) * 10 # 10배 스케일링

df = pd.DataFrame(X, columns=var, index=labels)
print(df)

# 데이터 시각화
plt.scatter(X[:, 0], X[:, 1], c='b', marker='o', s=50)
plt.grid()
plt.show()
plt.close()

# clustering 되는 모습 시각화 : 군집화, 최근접 탐색 전처리, 거리기반 시각화(히트맵), 이상치 탐지 등 용도
from scipy.spatial.distance import pdist, squareform
# pdist : 배열에 있는 값을 이용해 각 요소들의 거리 계산
# squareform : 거리 벡터를 사각형 형식(NxN)으로 변환
dist_vec = pdist(df, 'euclidean')
print(f'dist_vec : {dist_vec}')
'''
dist_vec : [5.3931329  1.38884785 4.89671004 2.40182631 5.09027885 7.6564396
 2.99834352 3.69830057 2.40541571 5.79234641]
'''

row_dist = pd.DataFrame(squareform(dist_vec), columns=labels, index=labels)
print(row_dist) # 어떤 값들 간의 거리값인지 알 수 있다.
'''
          점0        점1        점2        점3        점4
점0  0.000000  5.393133  1.388848  4.896710  2.401826
점1  5.393133  0.000000  5.090279  7.656440  2.998344
점2  1.388848  5.090279  0.000000  3.698301  2.405416
점3  4.896710  7.656440  3.698301  0.000000  5.792346
점4  2.401826  2.998344  2.405416  5.792346  0.000000
'''

# 응집형이란? 자료 하나하나를 군집으로 보고 가까운 군집끼리 연결하는 방법이다. 상향식이다
# 분리형이란? 전체 자료를 하나의 군집으로 보고 분리해 나가는 방법이다. 하향식이다.
# linkage : 응집형 계층적 군집을 수행한다. 
from scipy.cluster.hierarchy import linkage # 계층적 군집분석
row_clusters = linkage(dist_vec, method='ward')

# 군집으로 나눠졌다
df = pd.DataFrame(row_clusters, columns=['id_1','id_2','dist','mem_num']) # 
print(df)
'''
   id_1  id_2      dist  mem_num
0   0.0   2.0  1.388848      2.0
1   4.0   5.0  2.657109      3.0 # 점0 + 점2 = 점5, 점0,2,4 합쳤으므로 멤버수가 3
2   1.0   6.0  5.454004      4.0 # 점5 + 점4 = 점6
3   3.0   7.0  6.647102      5.0 # 점6 + 점3 = 점7
'''
# linkage의 결과로 덴드로그램 작성
from scipy.cluster.hierarchy import dendrogram
row_dendr = dendrogram(row_clusters, labels=labels) # row_clusters도 되고, df도 된다.
plt.tight_layout()
plt.ylabel('유클리드 거리')
plt.show()
plt.close()