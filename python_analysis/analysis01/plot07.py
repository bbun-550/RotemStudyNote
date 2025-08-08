'''
tips.csv로 요약 처리 후 시각화
'''
import pandas as pd
import matplotlib.pyplot as plt

tips = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tips.csv')

# tips.info()
# RangeIndex: 244 entries, 0 to 243
# Data columns (total 7 columns):
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   total_bill  244 non-null    float64
#  1   tip         244 non-null    float64
#  2   sex         244 non-null    object
#  3   smoker      244 non-null    object
#  4   day         244 non-null    object
#  5   time        244 non-null    object
#  6   size        244 non-null    int64
# dtypes: float64(2), int64(1), object(4)

tips['gender'] = tips['sex'] # gender 칼럼 생성
del tips['sex'] # sex 칼럼 삭제

# 팁 비율 구하기 : tip / total_bill
tips['tip_pct'] = tips['tip']/tips['total_bill'] # 파생 변수 생성
#    total_bill   tip    tip_pct
# 0       16.99  1.01   0.059447
# 1       10.34  1.66   0.160542

tip_pct_group = tips['tip_pct'].groupby([tips['gender'], tips['smoker']])
# groupby : 그룹별로 분할하여 독립된 그룹에 대하여 별도로 데이터를 처리(혹은 적용)하거나 그룹별 통계량을 확인하고자 할 때 사용
# print(tip_pct_group) # 객체 생성
# print(tip_pct_group.sum())
# print(tip_pct_group.max())
# print(tip_pct_group.min())

result = tip_pct_group.describe()
# print(result)

# print(tip_pct_group.agg('sum'))
# print(tip_pct_group.agg('mean'))
# print(tip_pct_group.agg('var'))

# 요약 통계로 시각화
def myFunc(group): # 사용자 정의 함수
    diff = group.max() - group.mean()
    return diff

result2 = tip_pct_group.agg(['var','mean','max','min',myFunc])
# print(result2)
#                     var      mean       max       min
# gender smoker
# Female No      0.001327  0.156921  0.252672  0.056797
#        Yes     0.005126  0.182150  0.416667  0.056433
# Male   No      0.001751  0.160669  0.291990  0.071804
#        Yes     0.008206  0.152771  0.710345  0.035638

result2.plot(kind='barh', title='agg func result')
# stacked=True : 누적 차트
plt.show()