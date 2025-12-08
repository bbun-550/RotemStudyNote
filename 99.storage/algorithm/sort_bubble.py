'''
버블 정렬 bubble sort
인접한 두 개의 원소를 비교하여 자리를 교환하는 방식
'''

# 방법 1
def bubble_sort(a):
    n = len(a)
    while True:
        changed = False # 자료를 바꾸었는지 여부
        for i in range(0, n - 1):
            if a[i] > a[i + 1]: # 앞이 뒤보다 크면
                a[i], a[i + 1] = a[i + 1], a[i]
                changed = True # 바뀜
            
        if changed == False:
            return

d = [2,4,5,1,3]
bubble_sort(d)
print(d)