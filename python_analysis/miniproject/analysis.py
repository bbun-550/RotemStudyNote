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


# print(np.unique(data.Income))
# print(min(data.Income)) # 1730.0
# print(max(data.Income)) # 666666.0

# print(min(data.Kidhome)) # 0
# print(max(data.Kidhome)) # 2

# 결측치 : Income에만 24개 있음
print(data.Income.isnull().sum()) # 24
print(data.Income.isna().sum()) # 24

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


x = data[['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 
          'MntSweetProducts','MntGoldProds']]

y_income = data['Income']
y_kid = data['Kidhome']

# 분포 확인 및 결측치, 이상치
import missingno as msno
msno.matrix(x, figsize=(12,5))
plt.show()
plt.close()

msno.matrix(data, figsize=(12,5))
plt.show()
plt.close()

plt.boxplot([x["MntWines"], x["MntFruits"], x["MntMeatProducts"],
             x["MntFishProducts"],x["MntSweetProducts"],x["MntGoldProds"]], 
            labels=["Wines", "Fruits", "Meat","Fish","Sweet","Gold"])
plt.show()
plt.close()
