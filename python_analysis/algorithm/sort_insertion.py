'''
삽입 정렬 insertion sort
자료 배열의 모든 요소를 앞에서부터 차례대로 이미 정렬된 배열 부분과 비교하여, 
자신의 위치를 찾아 삽입함으로써 정렬을 완성하는 알고리즘이다
'''

# 방법 1
def find_insFunc(r,v):
    # 이미 정렬된 r의 자료를 앞에서 부터 차례로 확인
    for i in range(0, len(r)):
        if v < r[i]: # 
            return i
    return len(r) # v가 r의 모든 요소값 보다 클 경우에는 맨 뒤에 삽입

def ins_sort(a):
    result = []
    while a:
        value = a.pop(0)
        ins_idx = find_insFunc(result, value)
        result.insert(ins_idx, value) # 찾은 위치에 값을 삽입(이후 값은 밀려남)
        # insert는 특정 지점에, append 뒤에
        # print(result)
    return result

d = [2,4,5,1,3]
print(ins_sort(d))
print()

# 방법 2
def ins_sort2(a):
    n = len(a)
    for i in range(1, n): # 두 번째 값(인덱스 : 1)부터 마지막까지 차례대로 삽입할 대상을 선택. 1부터 n-1까지 반복
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key: # key값 보다 큰 값을 우측으로 밀기(참일 때)
            a[j+1] = a[j]
            j -= 1
        a[j+1] = key
        # print(a)


d = [2,4,5,1,3]
ins_sort2(d)
print(d)


