"""
Keras 관련 라이브러리 구성요소 설명
- keras.Input: 함수형 API에서 입력 텐서 정의(Sequential에선 보통 생략 가능).
- keras.models.Sequential: 층을 순차적으로 쌓는 가장 단순한 모델 컨테이너.
- keras.layers.Dense: 완전연결(FC) 레이어. units(노드 수), activation(활성화) 등 지정.
- keras.layers.Activation: 별도의 활성화 층(활성화를 Dense의 인자로 넣는 것과 동일 효과).
- keras.optimizers.(SGD/RMSprop/Adam): 최적화 알고리즘. 학습률 등 하이퍼파라미터 설정.

참고: 환경에 따라 `from tensorflow import keras` 또는 `from tensorflow.keras ...` 형태를 권장하기도 합니다.
"""

# Keras 모델로 논리회로 분류 모델 작성
import numpy as np
# from keras import Input            # 함수형 API 입력 텐서(예: Input(shape=(n_features,)))
from keras.models import Sequential # 순차적 모델 컨테이너
from keras.layers import Dense, Activation # 완전연결층, 활성화층
from keras.optimizers import SGD, RMSprop, Adam # 대표 최적화기들(경사하강, RMSprop, Adam)

# 1 데이터 생성 세트
x = np.array([[0,0],[0,1],[1,0],[1,1]]) # dense 입력 2차원 요구한다
y = np.array([[0], [1], [1], [1]]) # OR 예시

# 2 모델 구성
# 작성 방법 1
model = Sequential([                 # 순차(Sequential) 모델 컨테이너 생성
    Dense(input_shape=(2,)),         # 입력 특성 2개를 받는 Dense 층(가중치 W, 편향 b 생성)
    Dense(units=1),                  # 출력 노드 1개(로짓). 이진 분류에서 확률로 변환 전 단계
    Activation('sigmoid'),           # 활성화 함수: 시그모이드(0~1 확률로 매핑)
])

# 작성 방법 2
model = Sequential()
model.add(Dense(units=1, input_shape=(2,))) # 입력 2차원, 출력 1차원 Dense 층 추가
model.add(Activation('sigmoid'))           # 활성화 층 추가(이항분류: sigmoid, 다항분류: softmax)

# 3 모델 학습 과정 설정
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy']) # stochastic gradient descent 확률적 경사하강법
# linear - metrics=mse | classify - metrics=accuracy

# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='SGD(learning_rate=0.01)', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='SGD(learning_rate=0.01, momentum=0.9)', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='RMSprop(learning_rate=0.01)', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='Adam(learning_rate=0.01)', loss='binary_crossentropy', metrics=['accuracy'])

# 4 모델 학습시키기
model.fit(x=x, y=y, epochs=500, batch_size=1, verbose=0) # epochs : 학습횟수 / batch_size(default 32, 2의 제곱) : 크면 클수록 속도가 빠르다. eg. 1 할당시, 한 문제풀고 답 맞추기. 

# 5 모델 평가 : evaluate()
loss_matrix = model.evaluate(x, y) # model.complie에 metrics가 있어야 반환한다.
# print(loss_matrix)
# [0.42539477348327637, 0.75] loss, accuarcy
# [0.2523323893547058, 1.0] loss, accuarcy => epochs 500 결과

# 6 모델 사용하기
proba = model.predict(x, verbose=0)
# print(proba)
'''
[[0.6962712 ]
 [0.650237  ]
 [0.7519292 ]
 [0.71082735]]
# 그냥 print 입력하면 이항분류 하고 있는데, 확률값이 출력된다.
'''
pred = (proba > 0.5).astype('int32')
# print(pred.ravel()) # 그냥 출력하면 2차원 출력
'''
[0 1 1 1]
'''

# 7 모델 저장
# model.save('test.keras')




# 아래는 사용 예시(필요 시 주석 해제하여 실행)
# X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
# y = np.array([[0],[1],[1],[0]], dtype=np.float32)  # XOR 예시
#
# model = Sequential([
#     Dense(8, input_shape=(2,)),   # 입력 2차원 → 은닉노드 8개
#     Activation('relu'),
#     Dense(1),
#     Activation('sigmoid'),        # 이진 분류: 시그모이드
# ])
#
# opt = Adam(learning_rate=0.01)    # 최적화기 선택
# model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()                    # 모델 구조 확인
# model.fit(X, y, epochs=200, verbose=0)
# print('acc =', model.evaluate(X, y, verbose=0)[1])
