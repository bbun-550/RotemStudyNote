'''
## 밀도 기반 클러스터링 비모수적 알고리즘
- 밀도 기반 클러스터링 비모수 알고리즘입니다. 
    어떤 공간의 점 집합이 주어지면 밀집된 점( 주변 이웃이 많은 점 )을 그룹화하고 
    저밀도 영역에 홀로 있는 점(가장 가까운 이웃이 너무 멀리 떨어져 있는 점)을 이상치로 표시한다.
    DBSCAN은 가장 흔히 사용되고 인용되는 클러스터링 알고리즘 중 하나이다.
- KMeans보다 느린 단점이 있다.
'''
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

x, y = make_moons(n_samples=200, noise=0.05, random_state=0) # noise 표준편차
print(x[:10])
print(f'실제 군집 id:{y[:10]}') # [0 1 1 0 1 1 0 1 0 1]

plt.scatter(x[:, 0], x[:, 1])
plt.show()
plt.close()

# KMeans로 군집화
km = KMeans(n_clusters=2, random_state=0)
pred1 = km.fit_predict(x)
print(f'예측 군집 id : {pred1[:10]}') # [1 1 0 0 1 1 1 1 1 1]

# 시각화: 전달된 라벨(pr)을 사용하고, 선택적으로 중심(centers)을 표시
def plotResultFunc(x, pr, centers=None, title=None):
    unique_labels = np.unique(pr)

    for lab in unique_labels:
        mask = (pr == lab)
        if lab == -1:  # DBSCAN의 잡음 포인트
            plt.scatter(x[mask, 0], x[mask, 1], c='gray', marker='x', s=40, label='noise')
        else:
            plt.scatter(x[mask, 0], x[mask, 1], s=40, label=f'cluster{lab}')

    # 중심점 (KMeans 등에서만 표시)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], marker='+', s=120, c='black', label='centroid')

    if title:
        plt.title(title)

    plt.legend()
    plt.show()
    plt.close()

plotResultFunc(x, pred1, centers=km.cluster_centers_, title='KMeans')

# DBSCAN으로 군집화
dm = DBSCAN(eps=0.2, min_samples=5, metric='euclidean') # min_samples : 반경 최소 포인트 개수
pred2 = dm.fit_predict(x)

plotResultFunc(x, pred2, title='DBSCAN')

# 군집화 : 고객 세분화, 예상치 탐지, 추천 시스템 ... 등 효과적
