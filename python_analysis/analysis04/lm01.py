## 최소 제곱해를 선형 행렬 방정식으로 구하기

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

x = np.array([0,1,2,3])
y = np.array([-1,0.2,0.5,2.1])
# plt.scatter(x,y) # 데이터 분포 보기
# plt.show()
# plt.close()

A = np.vstack([x, np.ones(len(x))]).T
# print(A)
# .lstsq()가 요구하는 데이터 형태이다. 뒤에 1,1,1,1은 무의미한 데이터이다.
# [[0. 1.]
#  [1. 1.]
#  [2. 1.]
#  [3. 1.]]

import numpy.linalg as lin
# y = wx + b라는 1차 방정식의 w, b?
w, b = lin.lstsq(A,y,rcond=None)[0] # 최소제곱법 연산
# 최소제곱법 : 잔차 제곱의 총합이 최소가 되는 값을 얻을 수 있다
print(f'w(weight,기울기,slope) : {w}\nb(bias, 절편, 편향, intercept) : {b}')
# w : 0.9599999999999999
# b : -0.9899999999999993

# 단순 선형회귀 수식(모델) 완성 : y = 0.9599 * x + -0.9899

# 회귀직선 시각화
plt.scatter(x,y)
plt.plot(x, w*x+b,label='실제값')
plt.legend()
plt.show()
plt.close()

# 수식을 써서 예측값 얻기
print(w * 1 + b) # -0.02999(예측값) - 0.2(실제값) = 잔차, 오차, 손실, 에러