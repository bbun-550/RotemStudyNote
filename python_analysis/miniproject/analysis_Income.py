import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', family='applegothic') # Mac : applegothic , WIN : malgun gothic
plt.rcParams['axes.unicode_minus'] = False

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from statsmodels.formula.api import ols
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, mean_squared_error, explained_variance_score
import scipy.stats as stats
import statsmodels.api as sm

all_cols = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds","Income"]
cols = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]

data = pd.read_csv('filtered_MarketingData.csv')
print(data.head())

# 이상치 확인
# plt.boxplot([data[i] for i in cols], labels=["Wines","Fruits","Meat","Fish","Sweet","Gold"])
# plt.title('품목별 이상치 확인')
# plt.show()
# plt.close()

# plt.boxplot(data['Income'])
# plt.title('Income 이상치 확인')
# plt.show()
# plt.close()

data = data.copy()
for c in all_cols:
    q1, q3 = data[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    data[c] = data[c].clip(lower=lower, upper=upper)

# 이상치 제거 후 boxplot
# plt.boxplot([data[i] for i in cols], labels=["Wines","Fruits","Meat","Fish","Sweet","Gold"])
# plt.title('품목별 이상치 제거 후')
# plt.show()
# plt.close()

# plt.boxplot(data['Income'])
# plt.title('Income 이상치 확인')
# plt.show()
# plt.close()


x = data[['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 
          'MntSweetProducts','MntGoldProds']]

y = data['Income']

# print(x.describe())

# train/test 분리
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=12)

# 스케일링과 모델을 일체형으로 관리
lm = make_pipeline( # Income
    StandardScaler(),
    LinearRegression()
    )

lm.fit(x_train, y_train)
pred = lm.predict(x_test)

# 선형회귀분석 - 기존 가정 충족 조건
# 1. 선형성
residual = y_test - pred
sns.regplot(x=pred, y=residual, lowess=True, line_kws={'color':'red'}) # lowess 비모수적 추정
plt.plot([pred.min(), pred.max()], [0,0], '--', color='gray')
plt.title('선형성')
plt.show()
plt.close()

# 2. 정규성
# print(f'shapiro test : {stats.shapiro(residual).pvalue}')
# shapiro test : 1.227834026671067e-08
# 0.05 보다 작으므로 정규성은 만족하지 않는다.

# 3. 독립성
print(f'Durbin-Watson : {sm.stats.stattools.durbin_watson(residual):.4f}')
# Durbin-Watson : 1.9610
# 2에 근접하므로 유의미하다.

# 4. 등분산성
sr = stats.zscore(residual)
sns.regplot(x=pred, y=np.sqrt(abs(sr)), lowess=True, line_kws={'color':'red'})
plt.title('품목별(독립변수) 간 등분산셩')
plt.show()
plt.close()

# 5. 다중공선성
from statsmodels.stats.outliers_influence import variance_inflation_factor
imsidf = data[cols]
vifdf = pd.DataFrame()
vifdf['vif_value'] = [variance_inflation_factor(imsidf, i) for i in range(imsidf.shape[1])]
print(vifdf)
#    vif_value
# 0   3.084218
# 1   3.891251
# 2   4.913613
# 3   4.171190
# 4   3.928216
# 5   2.580046
# VIF 값 모두 10 미만 이므로 "MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds" 간에 다중공선성 문제가 발생하지 않았다.



def RegScoreFunc(y_true, y_pred): # r2_score, explained_variance_score, mean_squared_error 써보자
    print(f'R2_score(결정계수): {r2_score(y_true, y_pred):.4f}')
    print(f'설명분산점수: {explained_variance_score(y_true, y_pred):.4f}') # 실제값, 예측값 순
    print(f'MSE(평균제곱오차): {mean_squared_error(y_true, y_pred):.4f}') # 숫자가 작을수록 좋다. 하지만 기준이 없다.(유동적)

RegScoreFunc(y_test, pred)


'''
R2_score(결정계수): 0.6541
설명분산점수: 0.6556
MSE(평균제곱오차): 151223814.3083
'''