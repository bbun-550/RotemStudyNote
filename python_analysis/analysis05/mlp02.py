# MLP 실습 : 종양 데이터
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()
x = cancer['data']
y = cancer['target']

# train/test
x_train, x_test, y_train, y_test = train_test_split(x,y)

# 표준화
scaler = StandardScaler()
scaler.fit(x_train)
scaler.fit(x_test)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 모델
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), solver='adam', learning_rate_init=0.1, verbose=1)
mlp.fit(x_train, y_train)
pred = mlp.predict(x_test)
print(f'예측값 : {pred[:5]}')
print(f'실제값 : {y_test[:5]}')
print(f'acc : {accuracy_score(y_test, pred):.4f}') # acc : 0.9860