'''
선택정렬 sort selection
주어진 데이터 리스트에서 가장 작은 원소를 선택하여 맨 앞으로 가져옴

# 최소값 찾기 : 정렬되지 않은 부분에서 가장 작은 값 찾기
# 교환 : 찾은 최소값을 정렬되지 않은 부분의 맨 앞으로 이동
# 반복 : 정렬되지 않은 부분의 크기가 1이 될 때까지 위 과정 반복
'''

# 방법 1 : 원리 이해 우선(공간 복잡도 고려하지 않음)
def find_minFunc(a): # 최소값 인덱스 찾기
    n = len(a)
    min_idx = 0
    for i in range(1, n):
        if a[i] < a[min_idx]:
            min_idx = i
    return min_idx

def sel_sort(a):
    result = []
    while a: # 자료 있으면 True, None이면 False
        min_idx = find_minFunc(a)
        value = a.pop(min_idx)
        result.append(value) # append, extend, +=
    return result

d = [2,4,5,1,3]
# print(find_minFunc(d))
print(f'방법1 :{sel_sort(d)}')

# 방법 2 : 일반적 정렬 알고리즘 구사
# 공간 x, d 안에서 정렬
# 각 반복마다 가장 작은 값을 해당 집합 내의 맨 앞자리와 값을 바꾼다
def sel_sort2(a):
    n = len(a)
    for i in range(0,n-1): # 0부터 n-2까지 반복
        min_idx = i
        for j in range(i+1,n):
            if a[j] < a[min_idx]:                
                min_idx = j
        a[i], a[min_idx] = a[min_idx], a[i]
            

d = [2,4,5,1,3]
sel_sort2(d) # call by reference. d 주소를 함수로 넘김
print(f'방법2 :{d}')