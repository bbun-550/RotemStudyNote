'''
## 공분산 상관계수
- 두 변수의 패턴을 확인하기 위해 공분산을 사용한다. 단위 크기의 영향을 받는다.
- 상관계수 : 공분산을 표준화, -1 ~ 0 ~ 1
    - +-1에 근사하면 관계가 강하다.
'''
import numpy as np
# 공분산 : 패턴의 방향은 알겠으나 구체적인 크기를 표현은 곤란하다.
print(np.cov(np.arange(1,6),np.arange(2,7))) # 2.5 우상향
print(np.cov(np.arange(10,60,10),np.arange(20,70,10))) # 250
print(np.cov(np.arange(100,600,100),np.arange(200,700,100))) # 25000

print(np.cov(np.arange(1,6),(3,3,3,3,3))) # 0 두 변수는 관련이 없다.
print(np.cov(np.arange(1,6),np.arange(6,1,-1))) # -2.5 우하향
print('---------------------------------------')

x = [8,3,6,6,9,4,3,9,3,4]
print(f'x의 평균 : {np.mean(x)}') # 5.5
print(f'x의 분산 : {np.var(x)}') # 5.45, 분산 : 평균과의 거리와 관련이 있다.

y = [6,2,4,6,9,5,1,8,4,5]
print(f'y의 평균 : {np.mean(y)}') # 5.0
print(f'y의 분산 : {np.var(y)}') # 5.4

# 시각화
import matplotlib.pyplot as plt
# plt.scatter(x,y)
# plt.show()
# plt.close()

# 공분산
print(f'x, y 공분산 : {np.cov(x,y)}') # 5.2222 : x값과 y값이 정비례이다.
print(f'x, y 공분산 : {np.cov(x,y)[0,1]:.4f}') # 0행1열만 꺼내서 보자. 
print('---------------------------------------')
# 상관계수
print(f'x, y 상관계수 : {np.corrcoef(x,y)}') # 0.8664 두 변수 간의 관계가 매우 강하다.
print(f'x, y 상관계수 : {np.corrcoef(x,y)[0,1]:.4f}') # 0행1열만 꺼내서 보자. 

# 참고 : 비선형인 경우는 일반적인 상관계수 방법을 사용하면 안된다.
m = [-3,-2,-1,0,1,2,3]
n = [9,4,1,0,1,4,9]
# plt.scatter(m,n)
# plt.show()
# plt.close()
print(f'm, n 상관계수 : {np.corrcoef(m,n)[0,1]:.4f}') # 0.0000 무의미한 작업