'''
## 선형회귀분석 - ols 사용
- mtcars dataset을 사용
- 독립변수가 종속변수(mpg, 연비)에 영향을 미치는가?
'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api # 여기에 mtcars dataset 있다
import matplotlib.pyplot as plt
import seaborn as sns

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
# print(mtcars.shape) # (32, 11)
# print(mtcars.columns) # ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
# mtcars.info() # 결측치, 자료형 확인 -> mpg float64

# 1. 상관계수 확인
# print(mtcars.corr()) 


# 독립변수 : hp, wt 
print(np.corrcoef(mtcars.hp, mtcars.mpg)[0,1]) # -0.7761
print(np.corrcoef(mtcars.wt, mtcars.mpg)[0,1]) # -0.8676 둘 다 음의 상관관계가 뚜렷하다.

# 2. 시각화
'''
plt.scatter(mtcars.hp, mtcars.mpg) # hp가 mpg에 영향을 준다고 가정
plt.xlabel('hp')
plt.ylabel('mpg')
plt.show()
plt.close()
'''

# 단순선형회귀 : hp - mpg
result = smf.ols(formula='mpg ~ hp', data=mtcars).fit()
print(f'검정결과\n{result.summary()}')
print(f'결정계수 : {result.rsquared:.4f}') # 0.6024
print(f'pvalue : {result.pvalues.iloc[1]:.4f}') # pvalue : 0.0000 < 0.05 이므로 유의한 모델이다.
#                            OLS Regression Results
# ==============================================================================
# Dep. Variable:                    mpg   R-squared:                       0.602
# Model:                            OLS   Adj. R-squared:                  0.589
# Method:                 Least Squares   F-statistic:                     45.46
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           1.79e-07
# Time:                        12:14:15   Log-Likelihood:                -87.619
# No. Observations:                  32   AIC:                             179.2
# Df Residuals:                      30   BIC:                             182.2
# Df Model:                           1
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     30.0989      1.634     18.421      0.000      26.762      33.436
# hp            -0.0682      0.010     -6.742      0.000      -0.089      -0.048
# ==============================================================================
# Omnibus:                        3.692   Durbin-Watson:                   1.134
# Prob(Omnibus):                  0.158   Jarque-Bera (JB):                2.984
# Skew:                           0.747   Prob(JB):                        0.225
# Kurtosis:                       2.935   Cond. No.                         386.
# ==============================================================================
# # 수식 : 예측값 = -0.0682 * x + 30.0989

# 수식으로 마력수 110, 50에 대한 연비 예측하기
print(f'마력수 110 : {-0.0682 * 110 + 30.0989}')
print(f'마력수 50 : {-0.0682 * 50 + 30.0989}')
# 마력수 110 : 22.5969
# 마력수 50 : 26.6889

# predict 방법으로 결과 출력
print(f'마력수 110 : {result.predict(pd.DataFrame({'hp':[110]}))}')
print(f'마력수 50 : {result.predict(pd.DataFrame({'hp':[50]}))}')
# 마력수 110 : 0    22.59375
# dtype: float64
# 마력수 50 : 0    26.687447
# dtype: float64


# 다중선형회귀 : hp,wt - mpg
result2 = smf.ols(formula='mpg ~ hp + wt', data=mtcars).fit()
print(f'검정결과\n{result2.summary()}')
print(f'결정계수 : {result2.rsquared:.4f}') # 0.8268
print(f'pvalue : {result2.pvalues.iloc[1]:.4f}') # pvalue : 0.0015 < 0.05 이므로 유의한 모델이다.
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                    mpg   R-squared:                       0.827
# Model:                            OLS   Adj. R-squared:                  0.815
# Method:                 Least Squares   F-statistic:                     69.21
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           9.11e-12
# Time:                        12:21:00   Log-Likelihood:                -74.326
# No. Observations:                  32   AIC:                             154.7
# Df Residuals:                      29   BIC:                             159.0
# Df Model:                           2
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     37.2273      1.599     23.285      0.000      33.957      40.497
# hp            -0.0318      0.009     -3.519      0.001      -0.050      -0.013
# wt            -3.8778      0.633     -6.129      0.000      -5.172      -2.584
# ==============================================================================
# Omnibus:                        5.303   Durbin-Watson:                   1.362
# Prob(Omnibus):                  0.071   Jarque-Bera (JB):                4.046
# Skew:                           0.855   Prob(JB):                        0.132
# Kurtosis:                       3.332   Cond. No.                         588.
# ==============================================================================
# 수식 : 예측값 = -0.0318 * x1 + -3.8778 * x2 + 37.2273