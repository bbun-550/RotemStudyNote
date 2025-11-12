import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.rc('font', family='applegothic') # Mac : applegothic , WIN : malgun gothic
plt.rcParams['axes.unicode_minus'] = False

from statsmodels.formula.api import ols

data = pd.read_csv('marketing_campaign.csv', sep="\t")

'''
use_data = data[['NumDealsPurchases','Year_Birth']]
print(use_data.isnull().sum())

x = use_data['Year_Birth']
y = use_data['NumDealsPurchases']

print(np.corrcoef(x, y)[0, 1]) # 0.06084555

import statsmodels.formula.api as smf
result = smf.ols(formula='Year_Birth ~ NumDealsPurchases', data=use_data).fit()
print(f'검정결과\n{result.summary()}')
print(f'결정계수 : {result.rsquared:.4f}') # 0.0037
print(f'pvalue : {result.pvalues.iloc[1]:.4f}') # 0.0040
'''
# 나이와 할인 받아 구매한 횟수 간에 관계가 없다.


# 출생년도, 나이로 변경
today = datetime.now().year
data['Age'] = today - data['Year_Birth']

# print(data['Age'].describe())
# plt.boxplot(data['Age'])
# plt.title('Age 이상치 확인')
# plt.show()
# plt.close()

q1, q3 = data['Age'].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
data['Age'] = data['Age'].clip(lower=lower, upper=upper)
# plt.boxplot(data['Age'])
# plt.title('Age 이상치 제거 후')
# plt.show()
# plt.close()

# print(data['Age'].describe())
# use_data = data[['NumDealsPurchases','Year_Birth']]
# print(use_data.isnull().sum())

x = data[["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]]
y = data['NumDealsPurchases']

# print(np.corrcoef(x, y)[0, 1]) # 0.06084555

import statsmodels.formula.api as smf
col = 'MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds'
result = smf.ols(formula=f'NumDealsPurchases ~ {col}', data=data).fit()
print(f'검정결과\n{result.summary()}')
print(f'결정계수 : {result.rsquared:.4f}') # 0.0037
print(f'pvalue : {result.pvalues.iloc[1]:.4f}') # 0.0040

# bins = list(range(20, 100, 10))
# labels = [f"{b}" for b in bins[:-1]]
# data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
# print(data['AgeGroup'].head())

# 결측 제거
# 일원분산분석: NumDealsPurchases ~ C(AgeGroup)
# anova_data = data[['AgeGroup','NumDealsPurchases']].dropna()
# import statsmodels.api as sm
# model = ols('NumDealsPurchases ~ C(AgeGroup)', data=anova_data).fit()
# table = sm.stats.anova_lm(model, type=2)
# print(table)




# data = data[['Year_Birth', 'Education', 'Marital_Status', 'Income', 
#             'Kidhome','Teenhome', 'Recency', 'MntWines', 'MntFruits',
#             'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
#             'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
#             'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']]


'''
import missingno as msno
msno.matrix(data_csv, figsize=(12,5))
plt.show()
plt.close()
print(data_csv.head())
data_csv = data_csv.dropna().reset_index(drop=True)
'''

'''
mean_income = np.round(data_csv['Income'].mean(),0)
print(mean_income)
data_csv = data_csv.Income.fillna(mean_income)
'''

# data_csv = data[['MntWines', 'MntFruits',
#             'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
#             'MntGoldProds','Income', 'Kidhome',]]

# data_csv.to_csv('filtered_MarketingData.csv', index=False, encoding='utf-8')

'''
x = data[['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 
          'MntSweetProducts','MntGoldProds']]

y_income = data['Income']
y_kid = data['Kidhome']

# 분포 확인 및 결측치, 이상치
import missingno as msno
msno.matrix(data, figsize=(12,5))
plt.show()
plt.close()
'''
