'''
회귀분석 문제 2) 
testdata에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다. 
이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오.  수학점수를 종속변수로 하자.
  - 국어 점수를 입력하면 수학 점수 예측
  - 국어, 영어 점수를 입력하면 수학 점수 예측
'''
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='applegothic')
import scipy.stats as stats
import numpy as np
import statsmodels.formula.api as smf

raw2 = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv'
data2 = pd.read_csv(raw2)
# print(data2.head(2))

# 데이터 칼럼 확인
# print(data2.columns)

# 이상치 확인
# data2.info()

# 이상치 시각화
# plt.boxplot([data2.국어, data2.영어, data2.수학])
# plt.xticks([1,2,3], ['국어','영어','수학'])
# plt.show()
# plt.close()

'''
# ---------- linregress 사용 ---------------
# 독립변수
lang1 = data2.국어

# 종속변수
math1 = data2.수학

model1 = stats.linregress(lang1, math1)
# print(f'기울기 : {model1.slope:.4f}')
# print(f'절편 : {model1.intercept:.4f}')
# print(f'상관계수 R : {model1.rvalue:.4f}')
# print(f'p-value : {model1.pvalue:.4f}')
# print(f'표준오차 : {model1.stderr:.4f}')
# print(f'R2 결정계수 : {model1.rvalue**2:.4f}')
# 기울기 : 0.5705
# 절편 : 32.1069
# 상관계수 R : 0.7663 -> 국어 점수와 수학 점수는 양의 상관관계를 갖는다. 국어 점수가 높으면 수학 점수도 높다.
# p-value : 0.0001 < 0.05 이므로 통계적으로 유의하다.
# 표준오차 : 0.1128
# R2 결정계수 : 0.5872 -> 독립변수가 종속변수를 58% 정도 설명하고 있다. 국어 점수와 수학 점수는 58% 관련이 있다.

# lang_score = int(input('국어 점수 입력 : '))
# print(f'수학 점수 예측 결과 : {np.polyval([model1.slope, model1.intercept], np.array([lang_score]))}점')
# 국어 점수 입력 : 70
# 수학 점수 예측 결과 : [72.0454057]점

# ---------- ols 사용 ---------------
# 단순 선형회귀 
result1 = smf.ols(formula='수학 ~ 국어', data=data2).fit()
print(f'검정결과1\n{result1.summary()}')
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     수학   R-squared:                       0.587
# Model:                            OLS   Adj. R-squared:                  0.564
# Method:                 Least Squares   F-statistic:                     25.60
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           8.16e-05
# Time:                        12:28:48   Log-Likelihood:                -76.543
# No. Observations:                  20   AIC:                             157.1
# Df Residuals:                      18   BIC:                             159.1
# Df Model:                           1
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     32.1069      8.628      3.721      0.002      13.981      50.233
# 국어             0.5705      0.113      5.060      0.000       0.334       0.807
# ==============================================================================
# Omnibus:                        1.833   Durbin-Watson:                   2.366
# Prob(Omnibus):                  0.400   Jarque-Bera (JB):                0.718
# Skew:                          -0.438   Prob(JB):                        0.698
# Kurtosis:                       3.310   Cond. No.                         252.
# ==============================================================================
lang_score = int(input('국어 점수 입력 : '))
print(f'국어{lang_score}점 받았을 때 수학 점수 : {result1.predict(pd.DataFrame({'국어':[lang_score]}))}')

# ----------------------------
# 다중 선형회귀

result2 = smf.ols(formula='수학 ~ 국어 + 영어', data=data2).fit()
print(f'검정결과2\n{result2.summary()}')
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     수학   R-squared:                       0.659
# Model:                            OLS   Adj. R-squared:                  0.619 # 61.9%의 설명력을 가진다.
# Method:                 Least Squares   F-statistic:                     16.46
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           0.000105 # 모델 자체는 유의하다.
# Time:                        12:24:57   Log-Likelihood:                -74.617
# No. Observations:                  20   AIC:                             155.2
# Df Residuals:                      17   BIC:                             158.2
# Df Model:                           2
# Covariance Type:            nonrobust # 강력하지 않다.
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     22.6238      9.482      2.386      0.029       2.618      42.629
# 국어             0.1158      0.261      0.443      0.663      -0.436       0.667 # 국어는 독립변수로 유의하지 않다.
# 영어             0.5942      0.313      1.900      0.074      -0.066       1.254
# ==============================================================================
# Omnibus:                        6.313   Durbin-Watson:                   2.163
# Prob(Omnibus):                  0.043   Jarque-Bera (JB):                3.824
# Skew:                          -0.927   Prob(JB):                        0.148
# Kurtosis:                       4.073   Cond. No.                         412.
# ==============================================================================

lang2_score = int(input('국어 점수 입력 : '))
eng2_score = int(input('영어 점수 입력 : '))
print(f'국어{lang2_score}점, 영어{eng2_score}점 받았을 때 수학 점수\n{result2.predict(pd.DataFrame({'국어':[lang2_score], '영어':[eng2_score]}))}')
'''

'''
회귀분석 문제 3)
- kaggle.com에서 carseats.csv 파일 Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
- 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
- 회귀분석모형의 적절성을 위한 조건도 체크하시오.
- 완성된 모델로 Sales를 예측.
'''
url3 = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Carseats.csv'
data3 = pd.read_csv(url3)
# print(data3.head(3))
# data3.info() # 칼럼 자료형 확인
# print(data3.columns) 
# ['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price','ShelveLoc', 'Age', 'Education', 'Urban', 'US']

# 결측치 확인
# print(data3.isnull().sum())
# print(data3.isna().sum())

data3 = data3[['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'Age', 'Education']]
# print(data3.head())

# 상관계수 분석
# print(data3.corr())

# print(f'CompPrice 상관계수 : {np.corrcoef(data3.CompPrice, data3.Sales)[0,1]:.4f}')
# print(f'Income 상관계수 : {np.corrcoef(data3.Income, data3.Sales)[0,1]:.4f}')
# print(f'Advertising 상관계수 : {np.corrcoef(data3.Advertising, data3.Sales)[0,1]:.4f}')
# print(f'Population 상관계수 : {np.corrcoef(data3.Population, data3.Sales)[0,1]:.4f}') 
# print(f'Price 상관계수 : {np.corrcoef(data3.Price, data3.Sales)[0,1]:.4f}')
# print(f'Age 상관계수 : {np.corrcoef(data3.Age, data3.Sales)[0,1]:.4f}')
# print(f'Education 상관계수 : {np.corrcoef(data3.Education, data3.Sales)[0,1]:.4f}')

# CompPrice 상관계수 : 0.0641 제외
# Income 상관계수 : 0.1520 제외
# Advertising 상관계수 : 0.2695
# Population 상관계수 : 0.0505 제외
# Price 상관계수 : -0.4450
# Age 상관계수 : -0.2318
# Education 상관계수 : -0.0520 제외



# 종속변수 : Sales

# 독립변수 선정
column_select = "+".join(data3.columns.difference(['Sales','CompPrice','Income','Population','Education']))
result3 = smf.ols(formula='Sales ~ '+ column_select, data=data3).fit()
print(result3.summary())
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                  Sales   R-squared:                       0.360
# Model:                            OLS   Adj. R-squared:                  0.355
# Method:                 Least Squares   F-statistic:                     74.10
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           4.63e-38
# Time:                        13:08:27   Log-Likelihood:                -893.24
# No. Observations:                 400   AIC:                             1794.
# Df Residuals:                     396   BIC:                             1810.
# Df Model:                           3
# Covariance Type:            nonrobust
# ===============================================================================
#                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------
# Intercept      16.0035      0.719     22.266      0.000      14.590      17.417
# Advertising     0.1231      0.017      7.201      0.000       0.089       0.157
# Age            -0.0488      0.007     -6.931      0.000      -0.063      -0.035
# Price          -0.0580      0.005    -12.022      0.000      -0.068      -0.049
# ==============================================================================
# Omnibus:                        2.431   Durbin-Watson:                   1.954
# Prob(Omnibus):                  0.297   Jarque-Bera (JB):                2.489
# Skew:                           0.186   Prob(JB):                        0.288
# Kurtosis:                       2.893   Cond. No.                         821.
# ==============================================================================

filter_data3 = data3[['Sales', 'CompPrice', 'Income', 'Advertising', 'Price', 'Age']]
# column_select = "+".join(data3.columns.difference(['Sales']))
# result3 = smf.ols(formula='Sales ~ '+ column_select, data=data3).fit()
