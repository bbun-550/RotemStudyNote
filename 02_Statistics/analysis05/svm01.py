# SVM : 비확률적 이진 선형분류모델 작성 가능
# - 직선적 분류 뿐만 아니라 kernel 트릭을 이용해 비선형 분류도 가능하다
# - 커널 kernels : 선형분류가 어려운 저차원 자료를 고차원 공간으로 매핑해서 분류를 한다.
# LogisticRegression과 SVM으로 XOR 연산 처리 결과 분류 가능 확인
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics
import pandas as pd
import numpy as np

x_data = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0],
]
x_df = pd.DataFrame(x_data)
feature = np.array(x_df.iloc[:,0:2])
label = np.array(x_df.iloc[:,2])
print(feature)
print(label)

# model = LogisticRegression()
model = svm.SVC() # 차원을 늘려서 분류
model.fit(feature,label)
pred = model.predict(feature)
print(f'예측값 : {pred}')
print(f'실제값 : {label}')
print(f'정확도 : {metrics.accuracy_score(label, pred)}') # 정확도 : 0.5 선형으로 분류 안된다.