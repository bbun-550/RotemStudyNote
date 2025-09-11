"""
TensorFlow 기초 연습 스크립트

다음 내용을 간단 예제로 보여줍니다.
- 텐서 생성(0D/1D/2D)과 dtype
- 브로드캐스팅 연산
- 텐서/넘파이 상호운용, tf.print vs print
- TF2에서의 eager 모드와 저수준 Graph 객체 사용

각 블록 위에 설명 주석을 추가하고, 기존 주석은 정확성을 높이도록 보완했습니다.
"""

import tensorflow as tf

# 버전/하드웨어 확인(필요 시 주석 해제)
print(tf.__version__)
print('GPU', '사용가능' if tf.config.list_physical_devices('GPU') else '불가')

# 즉시 실행 모드 확인: TF2 기본값은 True입니다.
# print(f'즉시 실행모드 : {tf.executing_eagerly()}')

# Tensor 생성 예시 (스칼라/벡터/행렬)
# print(1, type(1))
# 0D: 스칼라
# print(tf.constant(1), type(tf.constant(1)))
# 1D: 길이 1의 벡터
# print(tf.constant([1]), type(tf.constant([1])))
# 2D: 1×1 행렬
# print(tf.constant([[1]]), type(tf.constant([[1]])))
'''
tf.Tensor(1, shape=(), dtype=int32) <class 'tensorflow.python.framework.ops.EagerTensor'>
tf.Tensor([1], shape=(1,), dtype=int32) <class 'tensorflow.python.framework.ops.EagerTensor'>
tf.Tensor([[1]], shape=(1, 1), dtype=int32) <class 'tensorflow.python.framework.ops.EagerTensor'>
'''

# 브로드캐스팅 예시: c의 shape은 (2,), d는 (1,)로 (2,)에 맞춰 확장됩니다.
a = tf.constant([1,2])
b = tf.constant([3,4])
c = a + b # type: ignore
d = tf.constant([3])
e = c + d
# print(e)  # tf.Tensor([7 9], shape=(2,), dtype=int32) : broadcasting

# 연산자(+)와 tf.add는 동등합니다. TF2에서는 둘 다 동일한 연산 그래프를 생성합니다.
f = tf.add(c, d)
# print(f)
# tf.Tensor([7 9], shape=(2,), dtype=int32)
# 참고: "add보다 +가 빠르다"는 일반적으로 사실이 아닙니다(동일 연산).

# 동일 결과를 만드는 여러 방법(텐서 생성/캐스팅)
# print(tf.convert_to_tensor(7, dtype=tf.float32))
# print(tf.cast(7, dtype=tf.float32))
# print(tf.constant(7.0))
# print(tf.constant(7, dtype=tf.float32))
# <Output> tf.Tensor(7.0, shape=(), dtype=float32)

# NumPy와 상호운용: NumPy 배열도 연산 시 자동으로 텐서로 변환됩니다.
import numpy as np
arr = np.array([1, 2])
# print(arr, type(arr))
tfarr = tf.add(arr, 5)  # NumPy → Tensor 자동 변환(dType은 상황에 따라 int64가 될 수 있음)
# print(tfarr)
# tf.Tensor([6 7], shape=(2,), dtype=int64)

# tf.print는 그래프 모드와 eager 모드 모두에서 실제 값 출력. print는 EagerTensor 표현을 출력합니다.
# tf.print(f'tf.print :', tfarr)          # 값 출력(그래프에서도 실행 시 값이 찍힘)
# print(f'print : {tfarr.numpy()}')        # eager에서만 .numpy()로 값 접근 가능
'''
tf.print : [6 7]
print : [6 7]
'''

# NumPy 연산으로 다시 되돌리기: eager 상황에서 텐서는 NumPy로 복사되어 계산됩니다.
# print(np.add(tfarr, 3))  # [ 9 10]
# 주의: 그래프 모드(세션 실행)에서는 위 방식이 동작하지 않습니다.

# 저수준 Graph API 사용 예시(TF1 스타일). TF2에서는 일반적으로 eager + tf.function 사용을 권장.
g1 = tf.Graph()  # 별도의 그래프 생성(작업 영역)
# with g1.as_default():
#     c1 = tf.constant(1, name='c_one')
#     # 그래프 컨텍스트에서 생성된 텐서는 심볼릭 텐서(실행 전 노드)입니다.
#     print(c1)
#     print(type(c1))
#     # op 정의(proto)를 확인할 수 있습니다.
#     print(c1.op.node_def)

'''
Tensor("c_one:0", shape=(), dtype=int32)
<class 'tensorflow.python.framework.ops.SymbolicTensor'>
name: "c_one"
op: "Const"
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 1
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
'''
