'''
표준편차, 분산
- 두 반의 시험 성적이 '평균이 같다고 해서 성적분포가 동일한가?'
    - 판단하기 위해서는 표준편차와 분산을 알아야 한다. 
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
# 목표 평균 ...
target_mean = 60
std_dev_small = 10
std_dev_large = 20 # 평균에서 떨어져 있음

class1_raw = np.random.normal(loc=target_mean, scale=std_dev_small, size=100)
class2_raw = np.random.normal(loc=target_mean, scale=std_dev_large, size=100)

# 평균값 보정
class1_adj = class1_raw - np.mean(class1_raw) + target_mean
class2_adj = class2_raw - np.mean(class2_raw) + target_mean

# 정수화, 범위 제한 
class1 = np.clip(np.round(class1_adj), 10, 100).astype(int)
class2 = np.clip(np.round(class2_adj), 10, 100).astype(int)
# print(f'class1 : {class1}')
# print(f'class2 : {class2}')

# 통계값 계산 : 평균, 표준편차, 분산
mean1, mean2 = np.mean(class1),np.mean(class2) 
std1, std2 = np.std(class1),np.std(class2)
var1, var2 = np.var(class1),np.var(class2)

print(f'1반 성적\n평균 : {mean1:.2f}\n표준편차 : {std1:.2f}\n분산 : {var1:.2f}\n') # \n{class1}
print(f'2반 성적\n평균 : {mean2:.2f}\n표준편차 : {std2:.2f}\n분산 : {var2:.2f}\n') # \n{class2}

# 평균만 보고 성적을 판단하면 안된다
# 표준편차와 분산을 보고 판단해야 한다
# 1반은 평균 근처에 골고루 분포되어 있고, 2반은 넓게 퍼져있다

df = pd.DataFrame({
    'Class':['1반']*100 + ['2반']*100,
    'Store':np.concatenate([class1, class2])
})
print(df.head(2))
#   Class  Store
# 0    1반     66
# 1    1반     60

print(df.tail(2))
#     Class  Store
# 198    2반     61
# 199    2반     37