'''
알고리즘 algorithm
정의 :
문제를 해결하기 위한 일련의 단계적 절차 또는 방법을 의미합니다. 
즉, 어떤 문제를 해결하기 위해 컴퓨터가 따라 할 수 있도록 구체적인 명령어들을 순서대로 나열한 것이라고 할 수 있습니다.
컴퓨터 프로그램을 만들기 위한 알고리즘은 계산 과정을 최대한 구체적이고 명료하게 작성해야 합니다. 

문제 -> 데이터 입력 -> 알고리즘(롤 베이스)으로 처리 -> 결과 출력
'''
# 문제1) 1 - 10(n)까지의 정수의 합 구하기
def totFunc(n): # O(n)
    tot = 0
    for i in range(1, n+1):
        tot += i
    return tot

print("방법 1:", totFunc(10))

def totFunc2(n): # O(1)
    return n * (n+1) // 2

print("방법 2 :", totFunc2(10))

'''
주어진 문제를 푸는 방법은 다양하다. 어떤 방법이 더 효과적인지 알아내는 것이 '알고리즘 분석'

'알고리즘 분석' 평가 방법 : 계산 복잡도 평가방법
1. 공간 복잡도 : 메모리 사용량 분석 (요즘 그닥 중요하지 않음)
2. 시간 복잡도 : 처리 시간을 분석 (중요)
    O표기법 Big-O notation 으로 평가
    빅오 표기법은 알고리즘의 효율성을 표기해주는 표기법
'''
# 문제2) 임의의 정수들 중 최대값 찾기
# 입력 : 숫자 n개를 가진 list
# 최대값 찾기
# 출력 : 숫자 n개 중 최대값

def findMaxFunc(a): # O(N)
    n = len(a) # 입력 크기
    max_v = a[0]
    for i in range(1,n):
        if a[i] > max_v:
            max_v = a[i] # 최대값 변경

    return f"문제2) 최대값 : {max_v}"

d = [17,92,11,33,55,7,26,42]
print(findMaxFunc(d))

# 최대값 위치 반환
def findMaxFunc2(a): # O(N)
    n = len(a) # 입력 크기
    max_v = 0
    for i in range(1,n):
        if a[i] > a[max_v]: # 부호 방향 바꾸면 최소값 구할 수 있음
            max_v = i # 최대값 변경

    return f"문제2-1) 최대값 위치: {max_v}"

d = [17,92,11,33,55,7,26,42]
print(findMaxFunc2(d))

# 문제3) 동명이인 찾기.
# n명의 사람 이름 중 동일한 이름을 찾아서 결과 출력
imsi = ['길동','순신','순신','길동']
imsi2 = set(imsi) # 시험에서 쓰면 불합격
imsi = list(imsi2)
print(imsi)

def findSameFunc(a): # O(N^2)
    n = len(a)
    result = set()
    for i in range(0 ,n-1): # 0부터 n-2까지 반복
        for j in range(i + 1, n):
            if a[i] == a[j]: # 이름이 같으면
                result.add(a[i])
    return f'문제3) 동명이인 : {result}'    


names = ['tom', 'jerry', 'mike','tom'] # 재귀 : 함수가 자기 자신을 호출 >> 
print(findSameFunc(names))

# 문제4) 팩토리얼
# 방법1 for
def factFunc(n): # O(N)
    imsi = 1
    for i in range(1, n+1):
        imsi = imsi * i    
    return f'문제4-1) 팩토리얼 for : {imsi}'
print(factFunc(5))

# 방법2 재귀 호출(자기가 자기를 호출)
# 빠져나갈 수 있는 값 먼저
def factFunc2(n): # O(N)
    if n <= 1: # 종료 조건
        return 1
    return n * factFunc2(n-1)

print(f'문제4-2) 팩토리얼 재귀 호출 : {factFunc2(4)}')
# merge, quick sort할 때 사용

# 재귀 연습1) 1부터 n까지의 합 구하기
def func(n): # O(N)
    if n <= 1: # 종료 조건
        return 1
    return n + func(n-1)

print(f'재귀 연습1) 1부터 n까지의 합 구하기 : {func(10)}')

# 재귀 연습2) 숫자 n개 중 최대값 구하기
def findMax(a, n): # O(N)        
    if n <= 0:
        return 0
    current_max = a[n-1]
    next_max = findMax(a, n-1) 
    if next_max > current_max:
        current_max = next_max
    return current_max

values = [7,9,15,42,33,22]
print(f'재귀 연습2) 숫자 n개 중 최대값 구하기 : {findMax(values, len(values))}')

'''
def find_max_index(a, index=0):
    if index == len(a) - 1:
        return a[index]
    rest_max = find_max_index(a, index + 1)
    return a[index] if a[index] > rest_max else rest_max

values = [7, 9, 15, 42, 33, 22]
print(find_max_index(values))  # 42
'''



