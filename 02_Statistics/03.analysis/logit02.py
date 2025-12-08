## 날씨 예보(강우 여부)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv')
# print(data.head(3))
# print(data.shape) # (366, 12)

# 데이터 가공 : 필요한 데이터만 추출
data2 = pd.DataFrame()
data2 = data.drop(['Date','RainToday'], axis=1)

# - Yes, No 더미화
data2['RainTomorrow'] = data2['RainTomorrow'].map({'Yes':1, 'No':0})
# print(data2.head(3))
# print(data2.RainTomorrow.unique()) # [1 0] 더미화 성공적

# 학습데이터 / 검정데이터 분리 : 오버피팅 방지(분리하는 목적 중 하나)
train, test = train_test_split(data2, test_size=0.3, random_state=42, shuffle=True) # shuffle=True default ; 시계열 데이터 분석할 때는 False
# print(train.shape, test.shape) # 366개가 7:3 비율로 분리됐다. (256, 10) (110, 10)

# print(data2.columns) 
# ['MinTemp', 'MaxTemp', 'Rainfall', 'Sunshine', 'WindSpeed', 'Humidity','Pressure', 'Cloud', 'Temp', 'RainTomorrow']
# 이 중 RainTomorrow만 종속변수이다.
col_select = "+".join(train.columns.difference(['RainTomorrow'])) # 독립변수 메서드에 입력 용이하게 만들기 위함.

# 분류모델
my_formula = 'RainTomorrow ~' + col_select # 수식 완성
# model = smf.glm(formula=my_formula, data=train, family=sm.families.Binomial()).fit() # fit() : 학습해라. 즉, 최소제곱법을 쓴다.
# conf_tab 보기 위해 로짓 사용
model = smf.logit(formula=my_formula, data=train).fit()

'''
print(model.summary())
print(model.params) # 회귀계수만 볼 수 있다.
                 Generalized Linear Model Regression Results
==============================================================================
Dep. Variable:           RainTomorrow   No. Observations:                  253
Model:                            GLM   Df Residuals:                      243
Model Family:                Binomial   Df Model:                            9
Link Function:                  Logit   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -72.927
Date:                Thu, 28 Aug 2025   Deviance:                       145.85
Time:                        11:54:43   Pearson chi2:                     194.
No. Iterations:                     6   Pseudo R-squ. (CS):             0.3186
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    219.3889     53.366      4.111      0.000     114.794     323.984
Cloud          0.0616      0.118      0.523      0.601      -0.169       0.293 v
Humidity       0.0554      0.028      1.966      0.049       0.000       0.111
MaxTemp        0.1746      0.269      0.649      0.516      -0.353       0.702 v
MinTemp       -0.1360      0.077     -1.758      0.079      -0.288       0.016
Pressure      -0.2216      0.052     -4.276      0.000      -0.323      -0.120
Rainfall      -0.1362      0.078     -1.737      0.082      -0.290       0.018
Sunshine      -0.3197      0.117     -2.727      0.006      -0.550      -0.090
Temp           0.0428      0.272      0.157      0.875      -0.489       0.575 v
WindSpeed      0.0038      0.032      0.119      0.906      -0.059       0.066 v
==============================================================================
- pvalue가 충족하지 않은 변수를 한 번에 다 빼는게 아니라, 하나 씩 빼보면서 pvalue 확인 후에 제외를 결정한다.
- 아마 기술이 있지 않을까?...

Intercept    219.388868
Cloud          0.061599
Humidity       0.055433
MaxTemp        0.174591
MinTemp       -0.136011
Pressure      -0.221634
Rainfall      -0.136161
Sunshine      -0.319738
Temp           0.042755
WindSpeed      0.003785
'''

# 예측값
print(f'예측값 : {np.rint(model.predict(test)[:10].values)}')
print(f'실제값 : {test['RainTomorrow'][:10].values}')
# 예측값 : [0. 0. 0. 0. 0. 0. 1. 1. 0. 0.]
# 실제값 : [0 0 0 0 0 0 1 1 0 0]
# 얼핏 보기에 100% 같아 보인다.

# 정확도를 확인해보자
# confusion matrix
conf_tab = model.pred_table()
print(f'conf_tab\n{conf_tab}') # GLM은 pred_table을 지원하지 않는다. 로짓에서만 썼다. 그래서 위에서 로짓으로 모델 만들어줬다.
# [[197.   9.]
#  [ 21.  26.]]
print(f'분류 정확도 : {(conf_tab[0][0]+conf_tab[1][1])/len(train)}')
# 분류 정확도 : 0.87109375
'''
'''
from sklearn.metrics import accuracy_score
# print(f'glm 분류 정확도(메소드) : {accuracy_score(test['RainTomorrow'], np.around(model.predict(test).values))}')
# 정확도 : 0.87271