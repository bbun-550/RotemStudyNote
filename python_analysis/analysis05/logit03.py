'''
## LogisticRegression 클래스 - 다항분류 가능
- 활성함수는 softmax(sigmoid 대신)
'''
from sklearn import datasets # iris 데이터
from sklearn.linear_model import LogisticRegression # 
import numpy as np

# iris dataset 불러오기
iris = datasets.load_iris()
# print(iris.DESCR) # iris dataset에 대한 설명

# print(iris.keys()) # 'data', 'target' ...
# print(iris.target)

x = iris['data'][:, [3]] # 2차원 - petal length
# y = iris['target'] # 1차원
y = (iris['target'] == 2).astype(np.int32)

log_reg = LogisticRegression().fit(x,y) # solver : lbfgs (softmax 사용)

x_new = np.linspace(1,3,1000).reshape(-1,1) # 새로운 예측값을 얻기 위해 독립변수 생성(난수 발생 linspace)
# print(x_new)
# 결정 간격 보여주기 위함
y_proba = log_reg.predict_proba(x_new)
# print(y_proba)

# 시각화
import matplotlib.pyplot as plt
plt.plot(x_new, y_proba[:, 1],'r-',label='virginica')
plt.plot(x_new, y_proba[:, 0],'b--',label='not virginica')
plt.legend()
plt.xlabel('prtal width')
plt.tight_layout()
plt.show()


