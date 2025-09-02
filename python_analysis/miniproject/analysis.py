import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('marketing_campaign.csv', sep="\t")
data = data[['Year_Birth', 'Education', 'Marital_Status', 'Income', 
            'Kidhome','Teenhome', 'Recency', 'MntWines', 'MntFruits',
            'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']]

data_csv = data[['MntWines', 'MntFruits',
            'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
            'MntGoldProds','Income', 'Kidhome',]]

data_csv = data_csv.dropna().reset_index(drop=True)
print(data_csv.head())
'''mean_income = np.round(data_csv['Income'].mean(),0)
print(mean_income)
data_csv = data_csv.Income.fillna(mean_income)'''

'''
import missingno as msno
msno.matrix(data_csv, figsize=(12,5))
plt.show()
plt.close()
'''

data_csv.to_csv('filtered_MarketingData.csv', index=False, encoding='utf-8')

# print(np.unique(data.Income))
# print(min(data.Income)) # 1730.0
# print(max(data.Income)) # 666666.0

# print(min(data.Kidhome)) # 0
# print(max(data.Kidhome)) # 2

# 결측치 : Income에만 24개 있음
# print(data.Income.isnull().sum()) # 24
# print(data.Income.isna().sum()) # 24

# print(data.Kidhome.isnull().sum()) # 
# print(data.Kidhome.isna().sum()) # 

# print(data.MntWines.isnull().sum()) # 
# print(data.MntWines.isna().sum()) # 

# print(data.MntFruits.isnull().sum()) # 
# print(data.MntFruits.isna().sum()) # 

# print(data.MntMeatProducts.isnull().sum()) # 
# print(data.MntMeatProducts.isna().sum()) # 

# print(data.MntFishProducts.isnull().sum()) # 
# print(data.MntFishProducts.isna().sum()) # 

# print(data.MntSweetProducts.isnull().sum()) # 
# print(data.MntSweetProducts.isna().sum()) # 

# print(data.MntGoldProds.isnull().sum()) # 
# print(data.MntGoldProds.isna().sum()) # 


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
cols = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
plt.boxplot([data_csv["MntWines"], data_csv["MntFruits"], data_csv["MntMeatProducts"],
             data_csv["MntFishProducts"],data_csv["MntSweetProducts"],data_csv["MntGoldProds"]], 
            labels=["Wines", "Fruits", "Meat","Fish","Sweet","Gold"])
plt.show()
plt.close()

def drop_outlier_iqr(df, column):
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return df[df[column].between(lower, upper)]

'''data_iqr1 = drop_outlier_iqr(data, 'MntWines')
print(data_iqr1.shape)
data_iqr2 = drop_outlier_iqr(data, 'MntFruits')
print(data_iqr2.shape)
data_iqr3 = drop_outlier_iqr(data, 'MntMeatProducts')
print(data_iqr3.shape)
data_iqr4 = drop_outlier_iqr(data, 'MntFishProducts')
print(data_iqr4.shape)
data_iqr5 = drop_outlier_iqr(data, 'MntSweetProducts')
print(data_iqr5.shape)
data_iqr6 = drop_outlier_iqr(data, 'MntGoldProds')
print(data_iqr6.shape)
'''

filtered = data.copy()
for c in cols:
    q1, q3 = filtered[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered[c] = filtered[c].clip(lower=lower, upper=upper)

plt.boxplot([filtered[c] for c in cols], labels=["Wines","Fruits","Meat","Fish","Sweet","Gold"])
plt.title("After winsorization (column-wise clip)")
plt.show()
plt.close()