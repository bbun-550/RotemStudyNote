# 차원축속(PCA - 주성분 분석) 
import numpy as np
import pandas as pd
# 독립변수 feature
x1 = [95, 91, 66, 94, 68]
x2 = [56, 27, 25, 1, 9]
x3 = [57, 34, 9, 79, 4]
x = np.stack((x1,x2,x3), axis=1) # axis=1 : 각 변수를 열로 합친다. / axis=0 : 각 변수를 행으로 합친다
x = pd.DataFrame(x, columns=['x1','x2','x3'])
'''
or
x = np.stack((x1,x2,x3), axis=0)
x = pd.DataFrame(x.T, columns=['x1','x2','x3'])
'''
# print(x)

# 표준화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_std = scaler.fit_transform(x)
# print(x_std) # 표준화된 값
# print(scaler.inverse_transform(x_std)) # 표준화 원복

# PCA 처리
from sklearn.decomposition import PCA
pca = PCA(n_components=2) # n_components 주성분 개수를 지정
# print(pca.fit_transform(x_std)) # 주성분 처리 결과
# print(pca.inverse_transform(pca.fit_transform(x_std)))
# print(scaler.inverse_transform(pca.inverse_transform(pca.fit_transform(x_std)))) # 원래 값과 유사한 값으로 돌아간다.

print('와인 데이터로 분류 연습(RandomForest) - PCA 전과 후로 나눠어 실습')
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
from sklearn.model_selection import train_test_split
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/wine.csv', header=None)
print(data[:2])
x= np.array(data.iloc[:, 0:12])
y = np.array(data.iloc[:,12])
# 1 - fixed acidity 2 - volatile acidity 3 - citric acid4 - residual sugar5 - chlorides6 - free sulfur dioxide7 - total sulfur dioxide8 - density9 - pH10 - sulphates11 - alcoholOutput variable (based on sensory data):12 - quality (score between 0 and 10)
# print(x[:2])
# print(y[:2], np.unique(y)) # 1 - red, 0 - white

# train/test
train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.2, random_state=1)
# print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # (5197, 12) (1300, 12) (5197,) (1300,)

# 모델
model = RandomForestClassifier(criterion='entropy', n_estimators=100).fit(train_x, train_y)
pred = model.predict(test_x)
print(f'예측값 : {pred[:3]}')
print(f'실제값 : {test_y[:3]}')
print(f'정확도 : {sklearn.metrics.accuracy_score(test_y, pred):.4f}') # 정확도 : 0.9946 > overfitting
print()

# PCA
pca = PCA(n_components=3) # 12 칼럼을 3개로 축소 : 
x_pca = pca.fit_transform(x)
print(x[:2])
print(x_pca[:2])

# train/test
train_x, test_x, train_y, test_y = train_test_split(x_pca,y, test_size=0.2, random_state=1)

# 모델
model2 = RandomForestClassifier(criterion='entropy', n_estimators=100).fit(train_x, train_y)
pred2 = model2.predict(test_x)
print(f'예측값2 : {pred2[:3]}')
print(f'실제값 : {test_y[:3]}')
print(f'정확도2 : {sklearn.metrics.accuracy_score(test_y, pred2):.4f}') # 정확도2 : 0.9438
