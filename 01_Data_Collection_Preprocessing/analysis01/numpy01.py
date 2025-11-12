# 기본 통계 함수를 직접 작성 : 평균, 분산, 표준편차

grades = [1,3,-2,4]

def grades_sum(grades):
    tot = 0
    for i in grades:
        tot += i
    return tot

# print(grades_sum(grades))

def grades_ave(grades):
    ave = grades_sum(grades) / len(grades)
    return ave

# print(grades_ave(grades))

def grades_variance(grades):
    ave = grades_ave(grades)
    var1 = 0
    for i in grades:
        var1 += (i-ave)**2 #편차 제곱
    return var1 / len(grades) 
    
# print(grades_variance(grades))

def grades_std(grades):
    return grades_variance(grades) ** 0.5

# print(grades_std(grades))
print('=' * 20)

import numpy as np
print('합 : ', np.sum(grades))
print('mean평균 : ', np.mean(grades)) # 산술평균
# print('average평균 : ', np.average(grades)) # 가중평균을 구할 수 있음

print('분산 : ', np.var(grades)) # R과 값이 다를 수 있음 ; 자유도 이슈
print('표준편차 : ', np.std(grades))
