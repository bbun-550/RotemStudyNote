# 배열 연산
import numpy as np

x = np.array([[1,2], [3,4]], dtype=np.float64)
y = np.arange(5,9).reshape(2,2)
y = y.astype(np.float64)
print(x, x.astype, x.dtype)
print(y, y.astype, y.dtype)

# 요소별 합
print(x + y) # python 제공 산술연산자
print(np.add(x,y)) # numpy add 메소드,함수
# np.subract/multiply/divide
# python 연산속도(느림) > numpy 연산속도(빠름)

import time
big_arr = np.random.rand(1000000)
start = time.time()
sum(big_arr) # python 내장함수
end = time.time()
print(f'sum() : {end - start:.6f}sec')

start = time.time()
np.sum(big_arr) # numpy 함수; 연산속도 빠름
end = time.time()
print(f'np.sum() : {end - start:.6f}sec')

# 요소별 곱
print(x)
print(y)
print(x * y)
print(np.multiply(x,y))

print(x.dot(y)) # 내적 연산


