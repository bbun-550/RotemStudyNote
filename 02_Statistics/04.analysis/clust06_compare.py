# iris dataset으로 지도학습(KNN) vs 비지도학습(KMeans)
import numpy as np
from sklearn.datasets import load_iris

iris_dataset = load_iris()
# 출력 확인
# print(iris_dataset['data'][:3])
# print(iris_dataset['feature_names'][:3])
# print(iris_dataset['target'][:3])
# print(iris_dataset['target_names'][:3])

# train/test split
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(iris_dataset.data, iris_dataset.target, test_size=0.25, random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape, ) # (112, 4) (38, 4) (112,) (38,)

# 모델
print('지도학습 : K-NN ------------')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knnModel = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean') # distance - euclidmean, uniform - minkowski
knnModel.fit(train_x, train_y) # feature, label(tag, target, class)

# 예측
pred_label = knnModel.predict(test_x)
print(f'예측값 :{pred_label}') # [1 0 2 1 1 0 ...
print(f'test acc : {np.mean(pred_label == test_y):.3f}')
print(f'acc : {accuracy_score(test_y, pred_label):.3f}')

# 새로운 데이터 분류
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])
print(knnModel.predict(new_input)) # 인덱스 값
print(knnModel.predict_proba(new_input)) # 확률값

dist, index = knnModel.kneighbors(new_input)
print(f'dist:{dist}\nindex:{index}')

# 모델
print('비지도학습 : KMeans(데이터에 정답(label)이 없는 경우) ------------')
from sklearn.cluster import KMeans
kmeansModel = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
kmeansModel.fit(train_x) # label 없음 
# print(kmeansModel.labels_) # 군집 출력 [1 1 2 2 2 ... 
print(f'0 cluster : {train_y[kmeansModel.labels_ == 0]}')
print(f'1 cluster : {train_y[kmeansModel.labels_ == 1]}') # 결과값으로 setosa 라고 짐작할 수 있다
print(f'2 cluster : {train_y[kmeansModel.labels_ == 2]}')

# 이번에는 클러스터링에서 새로운 데이터 분류
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])
clu_pred = kmeansModel.predict(new_input)
print(clu_pred) # [2]