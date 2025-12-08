# 단층 신경망(뉴런, 노드) - 퍼셉트론 Perceptron
# - input의 가중치 합에 임계값을 기준으로 2가지 output 중 하나를 출력하는 간단한 구조이다.
# 단층 신경망 - 논리회로 분류
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

feature = np.array([[0,0],[0,1],[1,0],[1,1]]) # sklearn이라 2차원
print(feature)
label = np.array([0,0,0,1]) # and 연산
label = np.array([0,1,1,1]) # or 연산 
label = np.array([0,1,1,0]) # xor 연산 
ml = Perceptron(max_iter=100, eta0=0.1).fit(feature,label) # eta0 : learning rate 학습량
print(ml)
pred = ml.predict(feature)
print(f'pred : {pred}')
print(f'acc : {accuracy_score(label, pred):.4f}')
