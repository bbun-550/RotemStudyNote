# 텐서플로우 연산자
import tensorflow as tf
import numpy as np

x = tf.constant(7)
y = tf.constant(3)

# 삼항연산자
result = tf.cond(x > y, lambda:tf.add(x,y), lambda:tf.subtract(x,y))
# tf.print(result)
# 10

"""
TensorFlow에서의 삼항연산자(조건식) 정리
- 파이썬의 삼항연산자: `a if cond else b`
- 텐서플로우의 그래프 친화적 삼항연산자: `tf.cond(pred, true_fn, false_fn)`
  - `pred`: 스칼라 불리언 텐서
  - `true_fn`/`false_fn`: 지연 함수(람다) — 선택된 분기만 실행됨
- 원소별(벡터/행렬) 조건 선택: `tf.where(pred, x, y)`
  - `pred`의 shape이 `x`, `y`와 브로드캐스트 가능해야 함
아래 코드에 상세 주석과 추가 예시를 포함했습니다.
"""

# case 조건
f1 = lambda:tf.constant(1)
# tf.print(f1()) # 1
f2 = lambda:tf.constant(1)
a = tf.constant(3)
b = tf.constant(4)
# tf.case 실행 순서
# - pred_fn_pairs의 (조건, 함수) 쌍을 앞에서부터 평가합니다.
# - True가 되는 첫 번째 조건의 함수를 실행하고 그 값을 반환합니다.
# - 여러 조건이 동시에 True일 수 있는데, 기본값(exclusive=False)에서는 '첫 True 분기'만 실행됩니다.
# - exclusive=True로 설정하면 정확히 하나의 조건만 True여야 하며, 2개 이상 True면 오류가 발생합니다.
# - 어떤 조건도 True가 아니면 default 함수가 실행되고, default가 없으면 오류입니다.
# - 각 분기의 반환 dtype/shape은 서로 호환되어야 합니다.
result2 = tf.case([(tf.less(a, b), f1)], default=f2) # tf.less(a,b): a<b면 f1 실행, 아니면 default
# tf.print(result2) # 1

# 관계연산 : boolean 0, 1 반환
# tf.print(tf.equal(1, 2))
# tf.print(tf.not_equal(1, 2))
# tf.print(tf.greater(1, 2))
# tf.print(tf.greater_equal(1, 2))
# tf.print(tf.less(1, 2))

# 논리연산
# tf.print(tf.logical_and(True, False)) # 0
# tf.print(tf.logical_or(True, False)) # 1
# tf.print(tf.logical_not(True)) # 0

# 유일 합집합
kbs = tf.constant([1,2,2,2,3])
val, idx = tf.unique(kbs)
# tf.print(val.numpy())
# tf.print(idx.numpy())
'''
array([1, 2, 3], dtype=int32)
array([0, 1, 1, 1, 2], dtype=int32)
'''

# reduce : 차원 축소
ar = [[1.,2.], [3,4]]
tf.print(tf.reduce_mean(ar).numpy())
tf.print(tf.reduce_mean(ar, axis=0).numpy()) # 열 기준
tf.print(tf.reduce_mean(ar, axis=1).numpy()) # 행 기준


'''
# ---------------------------------------------
# Reduce: 차원 축소 연산
# ---------------------------------------------
# reduce_* 계열은 지정한 축(axis) 방향으로 값을 집계해 차원을 줄입니다.
# - axis=None: 모든 축을 따라 집계 → 스칼라
# - axis=k: k번째 축을 집계하고 그 축을 제거
# - axis=[i,j]: 여러 축을 한 번에 집계
# - keepdims=True: 집계된 축을 크기 1로 유지(브로드캐스팅에 유용)
# 예제 데이터(2x2 행렬)
ar = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
# 전체 평균(모든 요소) → 스칼라 2.5
# 참고: tf.print는 텐서를 바로 출력할 수 있으므로 .numpy()는 불필요합니다.
tf.print('mean(all):', tf.reduce_mean(ar))
# 축 기준 평균
# axis=0: 행을 따라 집계하여 열별 평균(열 기준) → [2., 3.]
tf.print('mean(axis=0):', tf.reduce_mean(ar, axis=0))

# axis=1: 열을 따라 집계하여 행별 평균(행 기준) → [1.5, 3.5]
tf.print('mean(axis=1):', tf.reduce_mean(ar, axis=1))

# keepdims=True: 축소된 축을 크기 1로 보존 → shape (2,1)
tf.print('mean(axis=1, keepdims=True):', tf.reduce_mean(ar, axis=1, keepdims=True))

# 다중 축 집계 예시(3D 텐서)
x3 = tf.reshape(tf.range(24, dtype=tf.float32), (2, 3, 4))  # shape (2,3,4)
tf.print('sum(axis=[1,2]):', tf.reduce_sum(x3, axis=[1, 2]))  # 결과 shape (2,)
# 정수형 평균 주의: 정수 dtype으로 mean을 하면 정수 나눗셈이 되어 절삭될 수 있습니다.
# 안전하게 실수로 캐스팅 후 사용 권장
ai = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
tf.print('int mean (casted):', tf.reduce_mean(tf.cast(ai, tf.float32)))

# 기타 대표적인 reduce 함수들
tf.print('sum:', tf.reduce_sum(ar))
tf.print('max:', tf.reduce_max(ar), 'min:', tf.reduce_min(ar))
tf.print('prod:', tf.reduce_prod(ar))

# 불리언 텐서의 축소: any/all
bools = tf.constant([[True, False], [True, True]])
tf.print('any:', tf.reduce_any(bools), 'all:', tf.reduce_all(bools))
'''
