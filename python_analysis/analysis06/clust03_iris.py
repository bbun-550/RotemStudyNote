# iris dataset을 이용한 계층적 군집분석
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(iris_df.head())
# print(iris_df.loc[0:4, ['sepal length (cm)', 'sepal width (cm)']])
'''
   sepal length (cm)  sepal width (cm)
0                5.1               3.5
1                4.9               3.0
2                4.7               3.2
3                4.6               3.1
4                5.0               3.6
'''

from scipy.spatial.distance import pdist, squareform
dist_vec = pdist(iris_df.loc[0:4, ['sepal length (cm)', 'sepal width (cm)']], # 꽃받침의 길이와 너비만 작업에 참여
            metric='euclidean')

print(f'dist_vec : {dist_vec}') # 어떤 점과 점 간의 거리인지 구분이 되지 않는다.

row_dist = pd.DataFrame(squareform(dist_vec))
print(row_dist) # 어떤 값들 간의 거리값인지 알 수 있다.

# 덴도그램으로 시각화
from scipy.cluster.hierarchy import linkage, dendrogram
row_clusters = linkage(dist_vec, method='complete') # complete : 완전 연결, ward : , single :
print(f'row_cluster :\n{row_clusters}')
'''
row_cluster :
[[0.         4.         0.14142136 2.        ]
 [2.         3.         0.14142136 2.        ]
 [1.         6.         0.31622777 3.        ]
 [5.         7.         0.64031242 5.        ]]
'''

## df에 넣어서 작업하자
df = pd.DataFrame(row_clusters, columns=['id1','id2','dist','mem_num'])
print(df)
'''
   id1  id2      dist  mem_num
0  0.0  4.0  0.141421      2.0
1  2.0  3.0  0.141421      2.0
2  1.0  6.0  0.316228      3.0
3  5.0  7.0  0.640312      5.0
'''

# row_dend = dendrogram(row_clusters)
# plt.tight_layout()
# plt.ylabel('dist')
# plt.show()
# plt.close()

from sklearn.cluster import AgglomerativeClustering # cluster 정보 볼 때 사용
ac = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='complete')
X = iris_df.loc[:, ['sepal length (cm)', 'sepal width (cm)']]
labels = ac.fit_predict(X)
print(f'클러스터 분류 결과 : {labels}')

plt.hist(labels)
plt.grid()
plt.show()