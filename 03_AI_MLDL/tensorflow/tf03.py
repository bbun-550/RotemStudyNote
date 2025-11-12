"""
TensorFlow 상수/변수와 제어흐름 기초

이 스크립트는 다음 개념을 간단 예제로 보여줍니다.
- tf.constant vs tf.Variable 차이와 갱신(assign)
- 텐서 산술 연산(tf.add, tf.multiply)과 dtype 승격
- 조건 분기(tf.cond)와 Autograph(@tf.function)로 그래프 변환
- tf.function 내부에서의 상태 업데이트(Variable)와 출력(tf.print)
"""

# constant / Variable 정리
import tensorflow as tf
import numpy as np

# constant(불변): 값 재할당 불가, 그래프/연산의 입력으로 사용
node1 = tf.constant(3, dtype=tf.float32)
node2 = tf.constant(4.0)
# print(node1)
# print(node2)

imsi = tf.add(node1, node2)
# print(imsi)

# Variable(가변): 학습 가능한 파라미터 등 상태(state)를 보관, assign으로 갱신 가능
node3 = tf.Variable(3, dtype=tf.float32)
node4 = tf.Variable(4.0)
print(node3.numpy())
print(node4)

imsi2 = tf.add(node3, node4)
# print(imsi2)
node4.assign(node3)
'''
tf.Tensor(3.0, shape=(), dtype=float32)
tf.Tensor(4.0, shape=(), dtype=float32)
tf.Tensor(7.0, shape=(), dtype=float32)
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=3.0>
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=4.0>
tf.Tensor(7.0, shape=(), dtype=float32)
'''

# 조건 분기: tf.cond(조건, 참-지연함수, 거짓-지연함수)
# 브랜치 함수는 lambda로 감싸서 필요할 때만 실행(지연 평가)됩니다.
a = tf.constant(5)
b = tf.constant(10)
c = tf.multiply(a, b)
result = tf.cond(a < b, lambda:tf.add(10,c), lambda:tf.square(a))
# print(result.numpy())
# 60

# 전역 Variable 상태를 tf.function에서 갱신하는 예제
v = tf.Variable(1)

@tf.function  # Autograph가 파이썬 if/연산을 그래프 연산(tf.cond 등)으로 변환합니다.
def find_nextFunc():
    v.assign(v + 1)
    # 텐서의 % 연산은 tf.math.floormod로 변환됩니다.
    if tf.equal(v % 2, 0):
        v.assign(v + 10)

find_nextFunc()
# print(v.numpy())
# 12

# 1부터 3까지 합 출력 함수 작성(단순 eager 실행)
def func():
    imsi = tf.constant(0)  # 스칼라 0과 동일하지만 텐서 객체
    su = 1
    for _ in range(3):
        imsi = tf.add(imsi, su)
        # imsi = imsi + su
        # imsi += su
        # 모두 동일 의미(덧셈)
    return imsi

kbs = func()
# print(kbs.numpy(), ' ', np.array(kbs))
# 3   3

# 파이썬 전역 이름 재바인딩 예제: 텐서는 불변이라 +=는 새 텐서를 만들어 이름에 재할당합니다.
imsi = tf.constant(0)

def func2():
    global imsi
    su = 1
    for _ in range(3):
        imsi += su
    return imsi

mbc = func2()
# print(mbc.numpy(), ' ', np.array(mbc))
# 3   3


# tf.function에서 상태 업데이트는 Variable로 수행해야 합니다.
imsi = tf.Variable(0)
@tf.function
def func3():
    # imsi = tf.Variable(0)  # 함수 내부에 두면 호출/트레이싱마다 새 변수 생성되어 비권장
    # 상태를 가지는 객체(값이 동적)
    su = 1
    for _ in range(3):
        # imsi += su  # 텐서처럼 재바인딩하는 대신, Variable은 assign_add로 상태를 갱신
        imsi.assign_add(su)  # Variable은 함수 밖(전역/객체 속성)에서 정의해야 안정적입니다.
    return imsi

sbs = func3()
# print(sbs.numpy(), ' ', np.array(sbs))
# 3   3


# 구구단 출력
@tf.function
def gugu(dan):
    su = tf.constant(0)
    for _ in range(9):
        su = tf.add(su, 1)
        # print(su)        
        # print(su.numpy)  # @ 선언 시, .numpy()는 트레이싱 중 호출 불가. 대신 tf.print 사용
        # tf.print(su)     # 위의 파이썬 print 대신 tf.print를 사용하면 그래프에서도 출력 가능
        
        # print(f'{dan}*{su}={dan * su:2}')  # 텐서를 f-string으로 포맷팅하면 그래프에서 오류
        tf.print(dan, '*', su, '=', dan*su)
        
        

# gugu(3)
'''
3*1= 3
3*2= 6
3*3= 9
3*4=12
3*5=15
3*6=18
3*7=21
3*8=24
3*9=27
'''

@tf.function
def gugu2(dan):
    for i in range(1, 10):
        result = tf.multiply(dan, i)  # multiply(): 원소(스칼라) 곱, 행렬 곱은 tf.matmul()
        tf.print(dan, '*', i, '=', dan*i)

gugu2(5)
'''
5 * 1 = 5
5 * 2 = 10
5 * 3 = 15
5 * 4 = 20
5 * 5 = 25
5 * 6 = 30
5 * 7 = 35
5 * 8 = 40
5 * 9 = 45
'''
        
