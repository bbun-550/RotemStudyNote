'''
회귀분석 문제 3)
- kaggle.com에서 carseats.csv 파일 Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
- 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
- 회귀분석모형의 적절성을 위한 조건도 체크하시오.
- 완성된 모델로 Sales를 예측.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False # 음수 깨짐 방지
import seaborn as sns
import statsmodels.formula.api as smf

url = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Carseats.csv'
df = pd.read_csv(url, usecols=[0,1,2,3,4,5,7,8])
# df = df.drop([df.columns[6],df.columns[9],df.columns[10]], axis=1)

# print(df.corr())
#                 Sales  CompPrice    Income  Advertising  Population     Price       Age  Education
# Sales        1.000000   0.064079  0.151951     0.269507    0.050471 -0.444951 -0.231815  -0.051955
# CompPrice    0.064079   1.000000 -0.080653    -0.024199   -0.094707  0.584848 -0.100239   0.025197
# Income v      0.151951  -0.080653  1.000000     0.058995   -0.007877 -0.056698 -0.004670  -0.056855
# Advertising v 0.269507  -0.024199  0.058995     1.000000    0.265652  0.044537 -0.004557  -0.033594
# Population   0.050471  -0.094707 -0.007877     0.265652    1.000000 -0.012144 -0.042663  -0.106378
# Price v      -0.444951   0.584848 -0.056698     0.044537   -0.012144  1.000000 -0.102177   0.011747
# Age v        -0.231815  -0.100239 -0.004670    -0.004557   -0.042663 -0.102177  1.000000   0.006488
# Education   -0.051955   0.025197 -0.056855    -0.033594   -0.106378  0.011747  0.006488   1.000000

lmodel = smf.ols(formula='Sales ~ Income + Advertising + Price + Age', data=df).fit()
# print(lmodel.summary())
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                  Sales   R-squared:                       0.371
# Model:                            OLS   Adj. R-squared:                  0.364
# Method:                 Least Squares   F-statistic:                     58.21
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           1.33e-38 < 0.05 유의한 모델
# Time:                        16:49:29   Log-Likelihood:                -889.67
# No. Observations:                 400   AIC:                             1789.
# Df Residuals:                     395   BIC:                             1809.
# Df Model:                           4
# Covariance Type:            nonrobust
# ===============================================================================
#                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------
# Intercept      15.1829      0.777     19.542      0.000      13.656      16.710
# Income          0.0108      0.004      2.664      0.008       0.003       0.019
# Advertising     0.1203      0.017      7.078      0.000       0.087       0.154
# Price          -0.0573      0.005    -11.932      0.000      -0.067      -0.048
# Age            -0.0486      0.007     -6.956      0.000      -0.062      -0.035
# ==============================================================================
# Omnibus:                        3.285   Durbin-Watson:                   1.931
# Prob(Omnibus):                  0.194   Jarque-Bera (JB):                3.336
# Skew:                           0.218   Prob(JB):                        0.189
# Kurtosis:                       2.903   Cond. No.                     1.01e+03
# ==============================================================================
# Income, Advertising, Price, Age 모두 < 0.05 

# 작성된 모델 저장 후 읽어서 사용한다. 모델 개발 후 다른 사람들은 lmodel.predict만 할 줄 알면 된다.
# 저장하는 방법
'''
# 1. pickle 모듈 사용
import pickle

# 저장
with open('mymodel.pickle', mode='wb') as obj:
    pickle.dump(lmodel, obj)

# 읽기
with open('mymodel.pickle', mode='rb') as obj:
    pickle.loads(obj)

mymodel.predict('~~')

# 2. joblib 모듈 사용
import joblib

# 저장
joblib.dump(lmodel, 'mymodel.model')

# 읽기
mymodel = joblib.load()
mymodel.predict('~~~')
'''

# 선형회귀분석의 기존 가정 충족 조건
df_lm = df.iloc[:,[0,2,3,5,6]]

# 잔차항 구하기
fitted = lmodel.predict(df_lm)
residual = df_lm.Sales - fitted # 잔차
# print(f'residual : {[residual[:3]]}')
# print(f'잔차 평균 : {[np.mean(residual)]}')

# 1. 선형성 : 잔차가 일정하게 분포되어야 한다.
'''
sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color':'red'}) # lowess 비모수적 추정
plt.plot([fitted.min(), fitted.max()], [0,0], '--', color='gray')
plt.show()
plt.close() # 잔차가 일정하게 분포하기 때문에 선형성 만족이다.
'''

# 2. 정규성 : 잔차항이 정규 분포를 따라야 한다.
# QQ plot 출력 다른 방법
'''
import scipy.stats as stats
sr = stats.zscore(residual)
[x,y],_ = stats.probplot(sr)
sns.scatterplot(x=x,y=y)
plt.plot([-3,3], [-3,3], '--', color='gray')
plt.show()
plt.close()
'''
# shapiro test
# print(f'shapiro test : {stats.shapiro(residual).pvalue:.4f}')
# shapiro test : 0.2127 > 0.05 이므로 정규성 만족

# 3. 독립성 : 독립변수의 값이 서로 관련되지 않아야 한다.
# Durbin-Watson: 1.931
# 2에 근사하고 있다. 자기상관이 없다.

import statsmodels.api as sm
# print(f'Durbin-Watson : {sm.stats.stattools.durbin_watson(residual):.4f}')
# Durbin-Watson : 1.9315

# 4. 등분산성

# 5. 다중공산성
