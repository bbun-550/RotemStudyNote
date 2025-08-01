import numpy as np

# step1 : array 관련 문제
# 정규분포를 따르는 난수를 이용하여 5행 4열 구조의 다차원 배열 객체를 생성하고, 각 행 단위로 합계, 최댓값을 구하시오.
# aa = np.random.randn(5,4)
# # print(aa)
# aa_sum = np.sum(aa, axis=1)
# aa_max = np.max(aa, axis=1)
# for i in range(5):
#     print(f'{i+1}행 합계 : {aa_sum[i]}')
#     print(f'{i+1}행 최댓값 : {aa_max[i]}')

# step2 : indexing 관련문제
# 2-1) 6행 6열의 다차원 zero 행렬 객체를 생성한 후 다음과 같이 indexing 하시오.
# aa = np.arange(1,37).reshape(6,6)
# zeros = np.zeros_like(aa)
# print(zeros)
# ## 조건1 > 36개의 셀에 1~36까지 정수 채우기
# bb = zeros + aa.reshape(6,6)
# print(bb)

# ## 조건2> 2번째 행 전체 원소 출력하기 
# print(bb[1])

# ## 조건3> 5번째 열 전체 원소 출력하기
# print(bb[:,4])

# ## 조건4> 15~29 까지 아래 처럼 출력하기
# print(bb[2:5,2:5])

# 2-2) 6행 4열의 다차원 zero 행렬 객체를 생성한 후 아래와 같이 처리하시오.
aa = np.arange(1,25).reshape(6,4)
zeros = np.zeros_like(aa)
print(f'1.zero 다차원 배열 객체\n{zeros}\n')

## 조건1> 20~100 사이의 난수 정수를 6개>>4개 발생시켜 각 행의 시작열에 난수 정수를 저장하고, 두 번째 열부터는 1씩 증가시켜 원소 저장하기
random_num = np.random.randint(20,101,6).reshape(6,1)
print(f'2.난수 정수 발생\n{random_num[:,0]}')

cc = zeros[0] + random_num
add_num = np.array([0,1,2,3])
dd = cc + add_num
print(f'3.zero 다차원 배열에 난수 정수 초기화 결과. 두 번째 열부터는 1씩 증가시켜 원소 저장하기\n{dd}')

## 조건2> 첫 번째 행에 1000, 마지막 행에 6000으로 요소값 수정하기
dd[0] = 1000
dd[5] = 6000
print(f'4.첫 번째 행에 1000, 마지막 행에 6000으로 수정\n{dd}')


# step3 : unifunc 관련문제
# 표준정규분포를 따르는 난수를 이용하여 4행 5열 구조의 다차원 배열을 생성한 후
#   아래와 같이 넘파이 내장함수(유니버설 함수)를 이용하여 기술통계량을 구하시오.
#   배열 요소의 누적합을 출력하시오.
 
# <<출력 예시>>
# ~ 4행 5열 다차원 배열 ~
# [[ 0.56886895  2.27871787 -0.20665035 -1.67593523 -0.54286047]
#            ...
#  [ 0.05807754  0.63466469 -0.90317403  0.11848534  1.26334224]] 

# ~ 출력 결과 ~
# 평균 :
# 합계 :
# 표준편차 :
# 분산 :
# 최댓값 :
# 최솟값 :
# 1사분위 수 :           percentile()
# 2사분위 수 :
# 3사분위 수 :
# 요소값 누적합 :      cumsum()



# numpy 문제 추가 ~~~~~~~~~~~~~~~~~~~~~
# Q1) 브로드캐스팅과 조건 연산
# 다음 두 배열이 있을 때,
# a = np.array([[1], [2], [3]])
# b = np.array([10, 20, 30])
# 두 배열을 브로드캐스팅하여 곱한 결과를 출력하시오.
# 그 결과에서 값이 30 이상인 요소만 골라 출력하시오.

a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])
result = a * b
print(result)

cond = np.where(result>=30)
print(result[cond])

# Q2) 다차원 배열 슬라이싱 및 재배열
#  - 3×4 크기의 배열을 만들고 (reshape 사용),  
#  - 2번째 행 전체 출력
#  - 1번째 열 전체 출력
#  - 배열을 (4, 3) 형태로 reshape
#  - reshape한 배열을 flatten() 함수를 사용하여 1차원 배열로 만들기

aa = np.arange(1,13).reshape(3,4)
print(aa[1])
# print(aa)
print(aa[:,0])
bb = aa.reshape(4,3)
print(bb)
flatten_bb = bb.flatten()
print(flatten_bb)

# Q3) 1부터 100까지의 수로 구성된 배열에서 3의 배수이면서 5의 배수가 아닌 값만 추출하시오.
# 그런 값들을 모두 제곱한 배열을 만들고 출력하시오.


# num3 = np.arange(3,101,3,)
# print(num3)

num3 = np.arange(3,101,3,dtype=int)
# num3 = np.delete(num3,4)
# print(num3[27]%15)


for i in range(len(num3)):
    if num3[i] % 3 == 0 and num3[i] % 5 != 0:
        num3_no5 = np.append(num3,i)
        print(num3)

print(num3)


# Q4) 다음과 같은 배열이 있다고 할 때,
# arr = np.array([15, 22, 8, 19, 31, 4])
# 값이 10 이상이면 'High', 그렇지 않으면 'Low'라는 문자열 배열로 변환하시오.
# 값이 20 이상인 요소만 -1로 바꾼 새로운 배열을 만들어 출력하시오. (원본은 유지)
# 힌트: np.where(), np.copy()





# Q5) 정규분포(평균 50, 표준편차 10)를 따르는 난수 1000개를 만들고, 상위 5% 값만 출력하세요.
# 힌트 :  np.random.normal(), np.percentile()



