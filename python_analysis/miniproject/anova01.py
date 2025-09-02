# 소득분위에 따른 고기, 와인 소비액 차이가 있다/없다(ANOVA)
#  - h0 : 소득분위에 따른 고기, 와인 소비량 차이가 없다
#  - h1 : 소득분위에 따른 고기, 와인 소비량 차이가 있다

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats
import seaborn as sns

plt.rc('font', family='applegothic')
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv('filtered_MarketingData.csv')

data = data[['MntMeatProducts','MntWines','Income']]

data.dropna(subset=['Income'], inplace=True)
bins = [0, 11600, 47150, np.inf]
labels = ['저소득층', '중산층', '고소득층']
data['Income_Group'] = pd.cut(data['Income'], bins=bins, labels=labels, right=True, include_lowest=True)
# print(data['Income_Group'].head())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# 왼쪽 그래프: 소득분위별 고기 소비량 박스플롯
sns.boxplot(x='Income_Group', y='MntMeatProducts', data=data, ax=axes[0])
axes[0].set_title('소득분위별 고기 소비량', fontsize=15)
axes[0].set_xlabel('소득분위', fontsize=12)
axes[0].set_ylabel('고기 소비량', fontsize=12)

# 오른쪽 그래프: 소득분위별 와인 소비량 박스플롯
sns.boxplot(x='Income_Group', y='MntWines', data=data, ax=axes[1])
axes[1].set_title('소득분위별 와인 소비량', fontsize=15)
axes[1].set_xlabel('소득분위', fontsize=12)
axes[1].set_ylabel('와인 소비량', fontsize=12)

plt.show()
plt.close()


# 이상치 제거
def invalid_num(column, k=1.5):
    q1, q3 = column.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return column.between(lower, upper)

valid = invalid_num(data['MntMeatProducts']) & invalid_num(data['MntWines'])
data = data.loc[valid].copy()


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# 왼쪽 그래프: 소득분위별 고기 소비량 박스플롯
sns.boxplot(x='Income_Group', y='MntMeatProducts', data=data, ax=axes[0],palette='flare')
axes[0].set_title('소득분위별 고기 소비량', fontsize=15)
axes[0].set_xlabel('소득분위', fontsize=12)
axes[0].set_ylabel('고기 소비량', fontsize=12)

# 오른쪽 그래프: 소득분위별 와인 소비량 박스플롯
sns.boxplot(x='Income_Group', y='MntWines', data=data, ax=axes[1],palette='flare')
axes[1].set_title('소득분위별 와인 소비량', fontsize=15)
axes[1].set_xlabel('소득분위', fontsize=12)
axes[1].set_ylabel('와인 소비량', fontsize=12)
plt.show()
plt.close()

groups_meat = [data['MntMeatProducts'][data['Income_Group'] == q] for q in ['저소득층', '중산층', '고소득층']]
groups_wines = [data['MntWines'][data['Income_Group'] == q] for q in ['저소득층', '중산층', '고소득층']]
f_stat_meat, p_value_meat = stats.f_oneway(*groups_meat)
f_stat_wines, p_value_wines = stats.f_oneway(*groups_wines)
print(f"고기 소비량(MntMeatProducts) 분석 결과: F-통계량 = {f_stat_meat:.4f}, p-value = {p_value_meat}")
print(f"와인 소비량(MntWines) 분석 결과: F-통계량 = {f_stat_wines:.4f}, p-value = {p_value_wines}")


# 정규성 검정
low_income = data[data['Income_Group']=='저소득층'] # 저소득층
mid_income = data[data['Income_Group']=='중산층'] # 중산층
high_income = data[data['Income_Group']=='고소득층'] # 고소득층

lincome_Meat = low_income['MntMeatProducts']
lincome_Wine = low_income['MntWines']
mincome_Meat = mid_income['MntMeatProducts']
mincome_Wine = mid_income['MntWines']
hincome_Meat = high_income['MntMeatProducts']
hincome_Wine = high_income['MntWines']

print(f'저소득-육류 정규성 : {stats.shapiro(lincome_Meat).pvalue}') 
print(f'저소득-주류 정규성 : {stats.shapiro(lincome_Wine).pvalue}')

print(f'중소득-육류 정규성 : {stats.shapiro(mincome_Meat).pvalue}')
print(f'중소득-주류 정규성 : {stats.shapiro(mincome_Wine).pvalue}')

print(f'고소득-육류 정규성 : {stats.shapiro(hincome_Meat).pvalue}')
print(f'고소득-주류 정규성 : {stats.shapiro(hincome_Wine).pvalue}')

# Kruskal-Wallis 검정
print("육류 소비량 Kruskal-Wallis:",
      stats.kruskal(lincome_Meat, mincome_Meat, hincome_Meat).pvalue)

print("와인 소비량 Kruskal-Wallis:",
      stats.kruskal(lincome_Wine, mincome_Wine, hincome_Wine).pvalue)


# 사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 육류 소비량
tukey_meat = pairwise_tukeyhsd(endog=data['MntMeatProducts'],
                               groups=data['Income_Group'],
                               alpha=0.05)
print("육류 소비량 Tukey HSD 결과:")
print(tukey_meat)

# 와인 소비량
tukey_wine = pairwise_tukeyhsd(endog=data['MntWines'],
                               groups=data['Income_Group'],
                               alpha=0.05)
print("와인 소비량 Tukey HSD 결과:")
print(tukey_wine)

# 시각화도
tukey_meat.plot_simultaneous(comparison_name='저소득층')
plt.title("육류 소비량 Tukey HSD 결과")
plt.show()

tukey_wine.plot_simultaneous(comparison_name='저소득층')
plt.title("와인 소비량 Tukey HSD 결과")
plt.show()