import pandas as pd
import numpy as np

data = pd.read_csv('marketing_campaign.csv', sep="\t")


data['all_purchase'] = (data['MntWines']) + (data['MntFruits']) + (data['MntMeatProducts']) + (data['MntFishProducts']) + (data['MntSweetProducts']) + (data['MntGoldProds'])
# print(data.all_purchase)
print()

use_data = data[['all_purchase','Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome']]
use_data['Education'] = use_data['Education'].map({'Basic':1, '2n Cycle':2, 'Graduation':3, 'Master':4, 'PhD':5})
use_data['Marital_Status'] = use_data['Marital_Status'].map({'Single':1, '2n Together':2, 'Married':3, 'Divorced':4, 'Widow':5, 'Alone':6, 'Absurd':7, 'YOLO':8})
pd.to_numeric(use_data['Income'], errors='coerce')
pd.to_numeric(use_data['Marital_Status'], errors='coerce')

use_data = use_data.dropna().reset_index(drop=True)

# print(use_data.info())
# print(use_data.head())

