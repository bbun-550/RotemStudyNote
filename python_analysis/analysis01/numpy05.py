# 배열에 행, 열 추가
import numpy as np

aa = np.eye(3) # 대각행렬
print('aa : \n', aa)
print()

aa = np.identity(3)

# 열
bb = np.c_[aa, aa[2]] # c_ ; column 추가
print(bb)

# 행
cc = np.r_[aa, [aa[0]]] # r_ ; row 추가
print(cc)
print()

# reshape
a = np.array([1,2,3])
print('np.c_ : \n', np.c_[a]) # 추가 값 안써주면 구조가 바뀜
# print(a) 바꾼 구조가 기억되지 않음
a.reshape(3,1)
print(a)

# append, insert, delete
## 1차원
print(a)
b = np.append(a, [4,5])
print(b)
c = np.insert(a, 0, [6,7]) # sources, loc, 삽입내용
print(c)
# d = np.delete(a, 1)
# d = np.delete(a, [1])
d = np.delete(a, [1,2])
print(d)
print()

## 2차원
aa = np.arange(1,10).reshape(3,3)
print(aa)
print(np.insert(aa,1,99)) # 상수 insert하니까 2차원 >> 1차원. 삽입 insert 후 차원 축소
print(np.insert(aa,1,99, axis=0)) # 1 행에 99로 채운다. 차원 변형 없음
print(np.insert(aa,1,99, axis=1)) # 1 열에 99로 채운다. 차원 변형 없음
print()

bb = np.arange(10,19).reshape(3,3)
print(bb)
cc = np.append(aa, bb) # 추가 append 후 차원 축소
print(cc)
cc = np.append(aa, bb, axis=0)
print(cc)
cc = np.append(aa, bb, axis=1)
print(cc)
print()

# np.append 연습
print(np.append(aa, 88))
print(np.append(aa, [[88,88,88]], axis=0))
print(np.append(aa, [[88],[88],[88]], axis=1))
print()

# np.delete 연습
print(np.delete(aa,1)) # 삭제 delete 후 차원 축소
print(np.delete(aa,1, axis=0))
print(np.delete(aa,1, axis=1))
print()
# c_ , r_ 보다 append, insert, delete가 명시적이다

# 조건 연산 : where(조건, 참, 거짓)
x = np.array([1,2,3])
y = np.array([4,5,6])
condData = np.array([True,False,True])
result = np.where(condData, x, y) # 참일 때 x 출력, 거짓일 때 y 출력
print(result)

aa = np.where(x >=2)
print(aa) # (array([1, 2]),) index 값이 출력
print(x[aa])
print(np.where(x >= 2, 'T', 'F'))
print(np.where(x >= 2, x, x+100))
print()

bb = np.random.randn(4, 4) # 정규분포(가우시안분포 - 중심극한정리) 난수에서 난수 출력해서 진정한 난수 출력
print(bb)
print(np.where(bb > 0, bb, 0))
print()

# 배열 결합 concatenate
kbs = np.concatenate([x,y])
print(kbs)

# 배열 분할
x1, x2 = np.split(kbs, 2)
print(x1, x2, sep='\n')
print()

a = np.arange(1,17).reshape(4,4)
print(a)
x1, x2 = np.hsplit(a, 2)
print(x1)
print(x2)

x1, x2 = np.vsplit(a, 2)
print(x1)
print(x2)
print()

# 복원, 비복원 추출
datas = np.array([1,2,3,4,5,6,7])

## 복원 ; 중복 O
for _ in range(5):
    print(datas[np.random.randint(0, len(datas)-1)], end=' ')
print()

### 복원 추출 함수 : choice()
print(np.random.choice(range(1,46), 6)) # [44 45 34 44 45  2]
### 비복원 추출 : choice() 
print(np.random.choice(range(1,46), 6,replace=False))
print()
### 비복원 추출(전용. random모듈에 있음) : sample()
import random
print(random.sample(datas.tolist(), 5)) # 5개만 출력

# 가중치를 부여한 램덤 추출
ar = 'air book cat d e f god'
ar = ar.split(' ')
print(ar)
print(np.random.choice(ar, 3, p=[0.1,0.1,0.1,0.1,0.1,0.1,0.4])) # p= 가중치
print()