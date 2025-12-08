"""
TensorFlow 변수와 연산 기초

이 스크립트는 다음 개념을 간단 예제로 보여줍니다.
- tf.Variable 생성과 shape/dtype
- 값 갱신: assign/assign_add/assign_sub
- Variable과 Tensor의 산술 연산 결과 차이
- @tf.function(Autograph)로 함수 그래프화
- 무작위 텐서 생성(uniform/normal)
- Eager 모드에서 .numpy()로 값 가져오기
"""

# tf에서 변수 선언 후 사용 예시
import tensorflow as tf

## Variable 생성
# 주의: 두 번째 위치 인자 `0`은 trainable=False로 해석됩니다. 가독성을 위해 키워드 사용 권장.
f = tf.Variable(1, 0)  # = tf.Variable(1, trainable=False)
t = tf.Variable(tf.ones((2,))) # 1D Tensor
m = tf.Variable(tf.ones((2,1))) # 2D Tensor
# print(f'f : {f}')
# print(f't : {t}')
# print(f'm : {m}')
# tf.print(m)
'''
f : <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=1>
t : <tf.Variable 'Variable:0' shape=(2,) dtype=float32, numpy=array([1., 1.], dtype=float32)>
m : <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
array([[1.],
       [1.]], dtype=float32)>
[[1]
 [1]]
'''

## 값 치환/갱신: .assign()
v1 = tf.Variable(1)
v1.assign(10) # 변수에 값 할당
# print(v1)
# <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=10>
# 스칼라

v2 = tf.Variable(tf.ones(shape=(1)))
v2.assign([20])
# print(v2)
# <tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([20.], dtype=float32)>
# 1차원

v3 = tf.Variable(tf.ones(shape=(1,2)))
v3.assign([[30, 40]])
# print(v3)
# <tf.Variable 'Variable:0' shape=(1, 2) dtype=float32, numpy=array([[30., 40.]], dtype=float32)>

## Variable 간/상수와의 연산: 결과는 일반적으로 Tensor(EagerTensor)
v1 = tf.Variable([3])
v2 = tf.Variable([5])
v3 = v1 * v2 + 10
# print(v3) 
# tf.Tensor([25], shape=(1,), dtype=int32)

var = tf.Variable([1,2,3,4,5], dtype=tf.float32)
result = var + 10
print(result)
# tf.Tensor([11. 12. 13. 14. 15.], shape=(5,), dtype=float32)

## 간단한 선형식 y = w*x + b
w = tf.Variable(tf.ones(shape=1,))
b = tf.Variable(tf.ones(shape=1,))
w.assign([2])
b.assign([3])

def func1(x):
    """
    일반 파이썬 함수로, Eager 모드에서 즉시 실행됩니다.
    내부의 w*x+b는 TensorFlow 연산이므로 텐서를 반환하지만,
    함수 자체는 파이썬 레벨에서 한 번씩 즉시 계산됩니다.
    """
    return w * x + b

@tf.function  # Autograph: 파이썬 함수를 그래프(tf.Graph)로 변환/추적하여 성능 향상한다.
def func2(x):
    return w * x + b

out_al = func1([3])
# print(f'out_al : {out_al}')
# print(type(func1))
'''
out_al : [9.]
<class 'function'>
'''

out_al = func2([1,2])
# print(f'out_al : {out_al}')
# print(type(func2))
'''
out_al : [5. 7.]
<class 'tensorflow.python.eager.polymorphic_function.polymorphic_function.Function'>
'''

## 난수 텐서 생성
rand = tf.random.uniform([4], 0, 1) # 균등 분포 U(0,1)
rand2 = tf.random.normal([4], 0, 1) # 정규 분포 N(0,1)
# print(f'균등 분포 : {rand}')
# print(f'정규 분포 : {rand2}')
'''
균등 분포 : [0.5902183 0.9601382 0.5428008 0.550123 ]
정규 분포 : [-0.12672856 -0.43997055  0.12908658  0.55270976]
'''

aa = tf.ones((2,1))
print(aa.numpy())
m = tf.Variable(tf.zeros((2,1)))
print(m.numpy())

m.assign(m.numpy()) # 치환: Eager에서 .numpy()로 값을 읽어와 다시 대입(실전에서는 직접 텐서 사용 권장)
print(m.numpy())

m.assign_add(aa) # 더하기 후 치환(in-place 업데이트)
print(m.numpy())

m.assign_sub(aa) # 빼기 후 치환(in-place 업데이트)
print(m.numpy())
