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
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from statsmodels.formula.api import ols
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, mean_squared_error

cols = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]

data = pd.read_csv('filtered_MarketingData.csv')
print(data.head())

data = data.copy()
for c in cols:
    q1, q3 = data[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    data[c] = data[c].clip(lower=lower, upper=upper)

plt.boxplot([data[i] for i in cols], labels=["Wines","Fruits","Meat","Fish","Sweet","Gold"])
plt.show()
plt.close()

x = data[['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 
          'MntSweetProducts','MntGoldProds']]

y = data['Kidhome']

print(x.describe())
'''
# train/test 분리
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=12)

# 스케일링과 모델을 일체형으로 관리
logi = make_pipeline( # Kidhome
    StandardScaler(),
    LogisticRegression(solver='lbfgs', max_iter=1000, random_state=12)
    )

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

tree = DecisionTreeClassifier(max_depth=5, random_state=12)

name_models = [('LR',logi),('KNN',knn),('DT',tree)]
for name, clf in name_models:
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print(f'{name} 정확도 : {accuracy_score(y_test, pred):.4f}')
'''

