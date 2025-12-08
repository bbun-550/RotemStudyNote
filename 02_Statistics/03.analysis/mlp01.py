# 다층 신경망(MLP) - 

# 다층 신경망 - 논리회로 분류
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

feature = np.array([[0,0],[0,1],[1,0],[1,1]]) # sklearn이라 2차원
print(feature)
label = np.array([0,0,0,1]) # and 연산
label = np.array([0,1,1,1]) # or 연산 
label = np.array([0,1,1,0]) # xor 연산 

# 방법 1 : hidden layer 하나에 노드 30개
ml = MLPClassifier(hidden_layer_sizes=30, solver='adam', learning_rate_init=0.01).fit(feature,label) # learning_rate : learning rate 학습량

# 방법 2 : hidden layer 3개 각각 노드 10개
ml = MLPClassifier(hidden_layer_sizes=(10,10,10), solver='adam', learning_rate_init=0.01).fit(feature,label) # learning_rate : learning rate 학습량

print(ml)
pred = ml.predict(feature)
print(f'pred : {pred}')
print(f'acc : {accuracy_score(label, pred):.4f}')
