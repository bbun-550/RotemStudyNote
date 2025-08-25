'''
## 단순선형회귀 - ols 사용
- 상관관계가 선형회귀모델에 미치는 영향
'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns # dataset 가지고 있음

iris = sns.load_dataset('iris')
# print(iris.head(3))
#    sepal_length  sepal_width  petal_length  petal_width species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa

# print(iris.corr()) # species 문자열이라 오류 발생
# print(iris.iloc[:,0:4].corr())
#               sepal_length  sepal_width  petal_length  petal_width
# sepal_length      1.000000    -0.117570      0.871754     0.817941
# sepal_width      -0.117570     1.000000     -0.428440    -0.366126
# petal_length      0.871754    -0.428440      1.000000     0.962865
# petal_width       0.817941    -0.366126      0.962865     1.000000

# 연습1 - 상관관계가 약한 두 변수(sepal_width, sepal_length)를 사용
result1 = smf.ols(formula='sepal_length ~ sepal_width', data=iris).fit() # formula 종속변수 ~ 독립변수
print(f'검정결과1\n{result1.summary()}')
print(f'결정계수 : {result1.rsquared:.4f}') # 0.0138
print(f'pvalue : {result1.pvalues.iloc[1]:.4f}') # pvalue : 0.1519 > 0.05 이므로 유의하지 않은 모델이다.
# --------------------------------검정결과1--------------------------------------
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:           sepal_length   R-squared:                       0.014 # 설명력 매우 떨어짐(좋은 모델 아님)
# Model:                            OLS   Adj. R-squared:                  0.007
# Method:                 Least Squares   F-statistic:                     2.074
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):              0.152 > 0.05
# Time:                        11:05:16   Log-Likelihood:                -183.00
# No. Observations:                 150   AIC:                             370.0
# Df Residuals:                     148   BIC:                             376.0
# Df Model:                           1
# Covariance Type:            nonrobust
# ===============================================================================
#                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------
# Intercept       6.5262      0.479     13.628      0.000       5.580       7.473
# sepal_width    -0.2234      0.155     -1.440      0.152      -0.530       0.083
# ==============================================================================
# Omnibus:                        4.389   Durbin-Watson:                   0.952
# Prob(Omnibus):                  0.111   Jarque-Bera (JB):                4.237
# Skew:                           0.360   Prob(JB):                        0.120
# Kurtosis:                       2.600   Cond. No.                         24.2
# ==============================================================================

# 시각화
'''
plt.scatter(iris.sepal_width, iris.sepal_length)
plt.plot(iris.sepal_width, result1.predict(), color='red')
plt.show()
plt.close()
'''

# 연습2 - 상관관계가 강한 두 변수(petal_length, sepal_length)를 사용
result2 = smf.ols(formula='sepal_length ~ petal_length', data=iris).fit() # formula 종속변수 ~ 독립변수
print(f'검정결과2\n{result2.summary()}')
print(f'결정계수 : {result2.rsquared:.4f}') # 
print(f'pvalue : {result2.pvalues.iloc[1]:.4f}') # 

# ----------------------------------검정결과2-----------------------------------
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:           sepal_length   R-squared:                       0.760 설명력 매우 높음
# Model:                            OLS   Adj. R-squared:                  0.758
# Method:                 Least Squares   F-statistic:                     468.6
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           1.04e-47 < 0.05 유의미한 결과이다
# Time:                        11:11:32   Log-Likelihood:                -77.020
# No. Observations:                 150   AIC:                             158.0
# Df Residuals:                     148   BIC:                             164.1
# Df Model:                           1
# Covariance Type:            nonrobust
# ================================================================================
#                    coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------
# Intercept        4.3066      0.078     54.939      0.000       4.152       4.462
# petal_length     0.4089      0.019     21.646      0.000       0.372       0.446
# ==============================================================================
# Omnibus:                        0.207   Durbin-Watson:                   1.867
# Prob(Omnibus):                  0.902   Jarque-Bera (JB):                0.346
# Skew:                           0.069   Prob(JB):                        0.841
# Kurtosis:                       2.809   Cond. No.                         10.3
# ==============================================================================

# 시각화
'''
plt.scatter(iris.petal_length, iris.sepal_length)
plt.plot(iris.petal_length, result2.predict(), color='red')
plt.show()
plt.close()
'''
print('-------------------------')
# 일부의 실제값과 예측값 비교
print(f'실제값 : {iris.sepal_length[:5].values}')
print(f'예측값 : {result2.predict()[:5]}')
# 실제값 : [5.1 4.9 4.7 4.6 5. ]
# 예측값 : [4.8790946  4.8790946  4.83820238 4.91998683 4.8790946 ]
# 95% 신뢰수준에서 76%의 설명력으로 도출된 결과이다.
print('-------------------------')

# 새로운 값으로 예측
new_data = pd.DataFrame({'petal_length':[1.1,0.5,5.0]})
y_pred = result2.predict(new_data)
print(f'예측 결과(sepal_length)\n{y_pred}')
# 0    4.756418 <- 1.1
# 1    4.511065 <- 0.5
# 2    6.351215 <- 5.0
# 꽃잎의 길이로 꽃받침의 길이를 예측했다.
# 상관계수가 1 또는 -1에 근사해야 좋은 모델이 나온다.

# 다중 선형회귀 맛보기 - 독립변수 복수
# result3 = smf.ols(formula='sepal_length ~ petal_length + petal_width + sepal_width', data=iris).fit() # formula 종속변수 ~ 독립변수

# 독립변수의 개수가 많을 때 쓸 수 있는 방법
column_select = "+".join(iris.columns.difference(['sepal_length','species']))
result3 = smf.ols(formula='sepal_length ~ '+ column_select, data=iris).fit()

print(f'검정결과3\n{result3.summary()}')
print(f'결정계수 : {result3.rsquared:.4f}') # 0.8586
print(f'pvalue : {result3.pvalues.iloc[1]:.4f}') # 0.0000
# -----------------------------------검정결과3-----------------------------------
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:           sepal_length   R-squared:                       0.859
# Model:                            OLS   Adj. R-squared:                  0.856 # 독립변수가 많아졌으므로 adj R 봐야한다.
# Method:                 Least Squares   F-statistic:                     295.5
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           8.59e-62 < 0.05 유의한 모델이다.
# Time:                        11:42:49   Log-Likelihood:                -37.321
# No. Observations:                 150   AIC:                             82.64
# Df Residuals:                     146   BIC:                             94.69
# Df Model:                           3
# Covariance Type:            nonrobust
# ================================================================================
#                    coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------
# Intercept        1.8560      0.251      7.401      0.000       1.360       2.352
# petal_length     0.7091      0.057     12.502      0.000       0.597       0.821
# petal_width     -0.5565      0.128     -4.363      0.000      -0.809      -0.304
# sepal_width      0.6508      0.067      9.765      0.000       0.519       0.783
# ==============================================================================
# Omnibus:                        0.345   Durbin-Watson:                   2.060
# Prob(Omnibus):                  0.842   Jarque-Bera (JB):                0.504
# Skew:                           0.007   Prob(JB):                        0.777
# Kurtosis:                       2.716   Cond. No.                         54.7
# ==============================================================================
