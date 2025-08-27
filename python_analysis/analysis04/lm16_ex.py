'''
다항회귀분석 문제) 
데이터 로드 (Servo, UCI) : "https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data"
cols = ["motor", "screw", "pgain", "vgain", "class"]

 - 타깃/피처 (숫자만 사용: pgain, vgain)
   x = df[["pgain", "vgain"]].astype(float)   
   y = df["class"].values
 - 학습/테스트 분할 ( 8:2 )
 - 스케일링 (StandardScaler)
 - 다항 특성 (degree=2) + LinearRegression 또는 Ridge 학습
 - 성능 평가 
 - 시각화
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data', header=None)
df.columns = ["motor", "screw", "pgain", "vgain", "class"]

# 데이터 확인
print(df.head())
df.info()

# train, test 데이터 분할
X = df[["pgain", "vgain"]].astype(float)
y = df["class"].values


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 스케일링
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 단순회귀
# model = LinearRegression()
# model.fit(x_train_scaled,y_train)

# print(x_train_scaled.shape)
# x_fit = x_train_scaled[:, np.newaxis]
# print(x_fit)

# y_pred = model.predict(x_test_scaled)
# model_r2 = r2_score(y_test, model.predict(x_test_scaled))
# # print(f'단순회귀 예측\n{y_pred}')
# print(f'model_r2 : {model_r2:.4f}') # 0.3380

# 다항(2차)회귀
quad = PolynomialFeatures(degree=2)
x_train_quad = quad.fit_transform(x_train_scaled)
x_test_quad = quad.transform(x_test_scaled)

# model = LinearRegression()
model = Ridge(alpha=1.0)
model.fit(x_train_quad, y_train)
y_quad_pre = model.predict(x_test_quad)
quad_r2 = r2_score(y_test, y_quad_pre)
# print(f'다항회귀 예측\n{y_quad_pre}')
print(f'quad_r2 : {quad_r2:.4f}') # 0.4117

# # 시각화
# x_fit = np.arange(X['pgain'].min(),X['pgain'].max(), 1)[:,np.newaxis]
# print(X.shape)
# print(x_train_quad)
print(y_quad_pre)


# plt.scatter(x_train['pgain'], y_train, label='학습데이터', color='lightgray')
