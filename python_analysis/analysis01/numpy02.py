# Numpy 기본 기능
import numpy as np

ss = ['tom', 'james', 'oscar', 5]
print(ss, type(ss)) # list ,로 구분

ss2 = np.array(ss) # ndarray 공백으로 구분
print(ss2, type(ss2)) # <class 'numpy.ndarray'>. 온갖 잡 type 다 담을 수 있음.
# type을 맞춰야 함. 다른 type 들어오면 알아서 맞춤. 동일 데이터 담고 있다
# int < float < complex <string 순

# 메모리 비교
li = list(range(1,9))
# print(li)
# print(id(li[0]), id(li[2]), sep=" ")
print(li*10)
print('=' * 40)
for i in li:
    print(i * 10, end=" ")

print('=' * 40)
print([i*10 for i in li])
print()

a = np.array([1,2,'3',0])
print(a, type(a), a.dtype, a.shape, a.ndim, a.size)
print(a[0], a[1])

b = np.array([[1,2,3], [4,5,6]])
print(b.shape, ' ', b[0], ' ', b[[1]]) # 2행 3열,, 차원 떨어뜨리는 방법
print(b[0,2], ' ', b[1,2])
print()

c = np.zeros((2,2))
print(c)
d = np.ones((2,2))
print(d)
e = np.full((3,3), fill_value=7)
print(e)
f = np.eye(9) # 단위행렬
print(f)
print()

print(np.mean(np.random.rand(500))) # 0-1 사이 난수 균등분포; 값들이 비이슷하게 나옴
print(np.mean(np.random.randn(500))) # 정규분포; 0의 근사

np.random.seed(40) # 40번째 난수표 사용할꺼야. 난수 고정
print(np.random.randn(2,3))
print()

# 인덱싱, 슬라이싱
a = np.array([1,2,3,4,5])
print(a)
print(a[1])
print(a[1:])
print(a[1:4])
print(a[1:5:2])
print(a[-2:])
print()

a = np.array([[1,2,3,4],[5,6,7,8],[10,11,12,13]])
print(a)
print(a[:])
print(a[1:])
print(a[1:, 0:2])
print(a[0], ' ', a[0][0], ' ', a[[[0]]])
print()

aa = np.array((1,2,3))
print(aa)
bb = aa[1:] # 서브 배열(논리적으로 )
print(bb)
bb[0] = 33
print(bb)
print(aa)
cc = aa[1:3].copy() # 복사본 생성
print(cc)
print(aa)
print()

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
r1 = a[1, :] # 1차원 배열 [4 5 6]
r2 = a[1:2, :] # 2차원 배열 [[4 5 6]]
print(r1, r1.shape)
print(r2, r2.shape)
print()

c1 = a[:, 1] # 전체열 1열만 슬라이싱
c2 = a[:, 1:2]
print(c1, c1.shape)
print(c2, c2.shape)
print()

print(a)
bool_idx = (a>=5) # 불리언으로 인덱싱 가능
print(a[bool_idx])