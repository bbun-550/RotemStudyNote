'''
퀵 정렬 quick sort
데이터를 정렬하는 대표적인 알고리즘 중 하나로, 분할 정복(Divide and Conquer) 방식
피벗(pivot)을 기준으로 작은 값은 왼쪽으로, 큰 값은 오른쪽으로 이동시켜 분할하고, 
각 부분 리스트를 재귀적으로 정렬하여 전체 리스트를 정렬하는 방식이다
'''

# 방법 1
def quick_sort(a):
    n = len(a)

    if n <= 1: # 리스트 요소가 하나면 정렬할 필요 없음. 종료 조건
        return a
    
    # 기준값 pivot 지정
    pivot = a[-1] # 일반적으로 제일 마지막 값을 기준값으로 지정. 사실 어떤 숫자를 지정해도 상관없음.
    g1 = []
    g2 = []

    for i in range(0, n-1):
        if a[i] < pivot:
            g1.append(a[i])
        else:
            g2.append(a[i])
    
    return quick_sort(g1) + [pivot] + quick_sort(g2)


d = [6,8,3,1,2,4,7,5]
print(quick_sort(d))
print()

# 방법 2 : 리스트 안에서 직접 정렬
def quick_sort_sub(a, start, end):    
    if end - start <= 0: # 종료 조건 : 정렬 대상이 한개 이하이면 정렬할 필요 없음
        return
    
    pivot = a[end]
    i = start
    for j in range(start, end):
        if a[j] < pivot:
            a[i], a[j] = a[j], a[i] # i 자리에 옮겨주고 i를 한 칸 뒤로 이동
            i += 1
    a[i], a[end] = a[end], a[i]

    quick_sort_sub(a, start, i - 1 ) # 왼쪽 부분 정렬
    quick_sort_sub(a, i + 1, end) # 오른쪽 부분 정렬


def quick_sort2(a):
    quick_sort_sub(a, 0,len(a)-1)

d = [6,8,3,1,2,4,7,5]
print(quick_sort2(d))