import pandas as pd
import numpy as np
from datetime import datetime


data = pd.read_csv('python_analysis/miniproject/marketing_campaign.csv', sep="\t")
# print(data.columns)
# data.info()

# data = data['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 
#             'Kidhome','Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 
#             'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
#             'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
#             'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
#             'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
#             'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response']

data = data[['Year_Birth', 'Education', 'Marital_Status', 'Income', 
            'Kidhome','Teenhome', 'Recency', 'MntWines', 'MntFruits',
            'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']]
# print(data.head(5))

# print(data['Education'].unique()) # ['Graduation' 'PhD' 'Master' 'Basic' '2n Cycle']
# print(data['Marital_Status'].unique()) # ['Single' 'Together' 'Married' 'Divorced' 'Widow' 'Alone' 'Absurd' 'YOLO']

today = datetime.now().year
data['Age'] = today - data['Year_Birth']
# print(data['Age'].head(5))

data['Education'] = data['Education'].map({'Basic':1, '2n Cycle':2, 'Graduation':3, 'Master':4, 'PhD':5})
data['Marital_Status'] = data['Marital_Status'].map({'Single':1, '2n Together':2, 'Married':3, 'Divorced':4, 'Widow':5, 'Alone':6, 'Absurd':7, 'YOLO':8})
pd.to_numeric(data['Income'], errors='coerce')
pd.to_numeric(data['Marital_Status'], errors='coerce')

data.info()
data = data.dropna().reset_index(drop=True)
'''
# MntWines
print(np.corrcoef(data['MntWines'], data['Income'])[0, 1]) # 0.68402926
print(np.corrcoef(data['MntWines'], data['Kidhome'])[0, 1]) # -0.4962969
print(np.corrcoef(data['MntWines'], data['Teenhome'])[0, 1]) # 0.0048464
print(np.corrcoef(data['MntWines'], data['Education'])[0, 1]) # 0.21407
print(np.corrcoef(data['MntWines'], data['Age'])[0, 1]) # 0.1577
print(np.corrcoef(data['MntWines'], data['Marital_Status'])[0, 1]) # 0.038879466
'''
'''
# MntFruits
print(np.corrcoef(data['MntFruits'], data['Income'])[0, 1]) # 0.50714821
print(np.corrcoef(data['MntFruits'], data['Kidhome'])[0, 1]) # -0.372581
print(np.corrcoef(data['MntFruits'], data['Teenhome'])[0, 1]) # -0.176763
print(np.corrcoef(data['MntFruits'], data['Education'])[0, 1]) # -0.0748622
print(np.corrcoef(data['MntFruits'], data['Age'])[0, 1]) # 0.0179172403
print(np.corrcoef(data['MntFruits'], data['Marital_Status'])[0, 1]) # 0.00535134
'''
'''
# MntMeatProducts
print(np.corrcoef(data['MntMeatProducts'], data['Income'])[0, 1]) # 0.6902572649
print(np.corrcoef(data['MntMeatProducts'], data['Kidhome'])[0, 1]) # -0.43712948
print(np.corrcoef(data['MntMeatProducts'], data['Teenhome'])[0, 1]) # -0.26115951
print(np.corrcoef(data['MntMeatProducts'], data['Education'])[0, 1]) # 0.04392617
print(np.corrcoef(data['MntMeatProducts'], data['Age'])[0, 1]) # 0.0308723
print(np.corrcoef(data['MntMeatProducts'], data['Marital_Status'])[0, 1]) # -0.0429015
'''
'''
# MntFishProducts
print(np.corrcoef(data['MntFishProducts'], data['Income'])[0, 1]) # 0.522099335
print(np.corrcoef(data['MntFishProducts'], data['Kidhome'])[0, 1]) # -0.38764395
print(np.corrcoef(data['MntFishProducts'], data['Teenhome'])[0, 1]) # -0.2041873
print(np.corrcoef(data['MntFishProducts'], data['Education'])[0, 1]) # -0.100230
print(np.corrcoef(data['MntFishProducts'], data['Age'])[0, 1]) # 0.04162537
print(np.corrcoef(data['MntFishProducts'], data['Marital_Status'])[0, 1]) # 0.0062786182
'''
'''
# MntSweetProducts
print(np.corrcoef(data['MntSweetProducts'], data['Income'])[0, 1]) # 0.5313851420
print(np.corrcoef(data['MntSweetProducts'], data['Kidhome'])[0, 1]) # -0.3706730
print(np.corrcoef(data['MntSweetProducts'], data['Teenhome'])[0, 1]) # -0.16247511
print(np.corrcoef(data['MntSweetProducts'], data['Education'])[0, 1]) # -0.093419082
print(np.corrcoef(data['MntSweetProducts'], data['Age'])[0, 1]) # 0.01813260656
print(np.corrcoef(data['MntSweetProducts'], data['Marital_Status'])[0, 1]) # 0.016067557
'''
'''
# MntGoldProds
print(np.corrcoef(data['MntGoldProds'], data['Income'])[0, 1]) # 0.38440729
print(np.corrcoef(data['MntGoldProds'], data['Kidhome'])[0, 1]) # -0.349594
print(np.corrcoef(data['MntGoldProds'], data['Teenhome'])[0, 1]) # -0.02172526
print(np.corrcoef(data['MntGoldProds'], data['Education'])[0, 1]) # -0.0930747670
print(np.corrcoef(data['MntGoldProds'], data['Age'])[0, 1]) # 0.061818181674
print(np.corrcoef(data['MntGoldProds'], data['Marital_Status'])[0, 1]) # 0.04110706
'''
