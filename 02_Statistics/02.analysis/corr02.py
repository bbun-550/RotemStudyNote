## 공분산과 상관계수 연습
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic') # 한글 깨짐 방지

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinking_water.csv')
print(data.head(3))
print(data.describe()) # 요약 통계량
print()
print(np.std(data.친밀도)) # 0.9685
print(np.std(data.적절성)) # 0.8580
print(np.std(data.만족도)) # 0.8271

# 시각화 히스토그램
# plt.hist([np.std(data.친밀도),np.std(data.적절성),np.std(data.만족도)])
# plt.show()
# plt.close()

print('---------공분산')
# numpy 공분산 출력 : 두 개만 확인 할 수 있다.
print(np.cov(data.친밀도, data.적절성))
print(np.cov(data.친밀도, data.만족도))
print(np.cov(data.적절성, data.만족도))
# pandas DataFrame 공분산 출력
print(data.cov())
#           친밀도       적절성       만족도
# 친밀도  0.941569  0.416422  0.375663
# 적절성  0.416422  0.739011  0.546333
# 만족도  0.375663  0.546333  0.686816

print('---------상관계수')
# numpy 상관계수 출력 : 두 개만 확인 할 수 있다.
print(np.corrcoef(data.친밀도, data.적절성))
print(np.corrcoef(data.친밀도, data.만족도))

# pandas DataFrame 상관계수 출력
print(data.corr()) # pearson이 기본값.
#           친밀도       적절성       만족도
# 친밀도  1.000000  0.499209  0.467145
# 적절성  0.499209  1.000000  0.766853
# 만족도  0.467145  0.766853  1.000000

print(data.corr(method='pearson')) # 변수가 등간, 비율 척도일 때
print(data.corr(method='spearman')) # 변수가 서열 척도일 때
print(data.corr(method='kendall')) # spearman의 다른 형태

# 예) 만족도에 대한 다른 특성(변수) 사이의 상관관계 보기
co_re = data.corr()
print(co_re['만족도'].sort_values(ascending=False)) # descending sort 출력 방식
# 만족도    1.000000
# 적절성    0.766853
# 친밀도    0.467145

# 상관관계 시각화
data.plot(kind='scatter', x='만족도', y='적절성')
plt.show()
plt.close()

from pandas.plotting import scatter_matrix
attr = ['친밀도','적절성','만족도']
scatter_matrix(data[attr], figsize=(10,6))
plt.show()
plt.close()

# heatmap
# 색이 진할수록 약한 상관관계, 연할수록 강한 상관관계
import seaborn as sns
sns.heatmap(data.corr())
plt.show()
plt.close()
