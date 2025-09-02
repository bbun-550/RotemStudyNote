import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from statsmodels.formula.api import ols
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, mean_squared_error


data = pd.read_csv('filtered_MarketingData.csv')
print(data.head())

x = data[['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 
          'MntSweetProducts','MntGoldProds']]

y = data['Income']

# train/test 분리
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=12)

# 스케일링과 모델을 일체형으로 관리
lm = make_pipeline( # Income
    StandardScaler(),
    LinearRegression()
    )

