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
sns.boxplot(x='Income_Group', y='Income', data=data, ax=axes[0])
axes[0].set_title('소득분위별 고기 소비량', fontsize=15)
axes[0].set_xlabel('소득분위', fontsize=12)
axes[0].set_ylabel('소득', fontsize=12)

# 오른쪽 그래프: 소득분위별 와인 소비량 박스플롯
sns.boxplot(x='Income_Group', y='Income', data=data, ax=axes[1])
axes[1].set_title('소득분위별 와인 소비량', fontsize=15)
axes[1].set_xlabel('소득분위', fontsize=12)
axes[1].set_ylabel('소득', fontsize=12)

plt.show()
plt.close()


# 이상치 제거
def invalid_num(column, k=1.5):
    q1, q3 = column.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return column.between(lower, upper)

valid = invalid_num(data['Income'])
data = data.loc[valid].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# 왼쪽 그래프: 소득분위별 고기 소비량 박스플롯
sns.boxplot(x='Income_Group', y='Income', data=data, ax=axes[0],palette='flare')
axes[0].set_title('소득분위별 고기 소비량', fontsize=15)
axes[0].set_xlabel('소득분위', fontsize=12)
axes[0].set_ylabel('소득', fontsize=12)

# 오른쪽 그래프: 소득분위별 와인 소비량 박스플롯
sns.boxplot(x='Income_Group', y='Income', data=data, ax=axes[1],palette='flare')
axes[1].set_title('소득분위별 와인 소비량', fontsize=15)
axes[1].set_xlabel('소득분위', fontsize=12)
axes[1].set_ylabel('소득', fontsize=12)

plt.show()
plt.close()