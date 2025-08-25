'''
## 선형회귀분석 - ols 사용
- 선형회귀분석의 기존 가정 충족 조건
- 선형성 : 독립변수(feature)의 변화에 따라 종속변수도 일정 크기로 변화해야 한다.
- 정규성 : 잔차항(오차항)이 정규분포를 따라야 한다.
- 독립성 : 독립변수의 값이 서로 관련되지 않아야 한다.
- 등분산성 : 그룹간의 분산이 유사해야 한다. 독립변수의 모든 값에 대한 오차들의 분산은 일정해야 한다.
- 다중공선성 : 다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다.
'''

# advertising.csv 사용 : 각 매체의 광고비에 따른 판매량
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False # 음수 깨짐 방지
import seaborn as sns
import statsmodels.formula.api as smf

advdf = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv',
                    usecols=[1,2,3,4])
# print(advdf.head(3))
# print(advdf.shape) # (200, 4)
# print(advdf.index, advdf.columns)
# advdf.info() # 모든 자료형 float64

# 상관관계 확인(상관계수)
# 한 방에 다 보기
# print(advdf.corr())
#                  tv     radio  newspaper     sales
# tv         1.000000  0.054809   0.056648  0.782224
# radio      0.054809  1.000000   0.354104  0.576223
# newspaper  0.056648  0.354104   1.000000  0.228299
# sales      0.782224  0.576223   0.228299  1.000000

# 단순 선형회귀 모델 tv - sale 
# x = tv, y = sales
lmodel1 = smf.ols(formula='sales ~ tv', data=advdf).fit()

print(lmodel1.params) # coef
print(lmodel1.pvalues)
print(lmodel1.rsquared) # 0.611875050850071

# print(lmodel1.summary())
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                  sales   R-squared:                       0.612
# Model:                            OLS   Adj. R-squared:                  0.610
# Method:                 Least Squares   F-statistic:                     312.1
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           1.47e-42
# Time:                        15:02:30   Log-Likelihood:                -519.05
# No. Observations:                 200   AIC:                             1042.
# Df Residuals:                     198   BIC:                             1049.
# Df Model:                           1
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      7.0326      0.458     15.360      0.000       6.130       7.935
# tv             0.0475      0.003     17.668      0.000       0.042       0.053
# ==============================================================================
# Omnibus:                        0.531   Durbin-Watson:                   1.935
# Prob(Omnibus):                  0.767   Jarque-Bera (JB):                0.669
# Skew:                          -0.089   Prob(JB):                        0.716
# Kurtosis:                       2.779   Cond. No.                         338.
# ==============================================================================

# print(lmodel1.summary().tables[0]) # 테이블 윗 부분
# print(lmodel1.summary().tables[1]) # 아래 부분

# 기존 자료로 예측하기
x_part = pd.DataFrame({'tv':advdf.tv[:3]})
# print(f'실제값 : {advdf.sales[:3]}')
# print(f'예측값 : {lmodel1.predict(x_part).values}') # [17.97077451  9.14797405  7.85022376]

# 새로운 자료로 예측하기
x_new = pd.DataFrame({'tv':[100,300,500]})
# print(f'새 자료 예측값 : {lmodel1.predict(x_new).values}') # [11.78625759 21.29358568 30.80091377]

# 시각화
'''
plt.scatter(advdf.tv, advdf.sales)
plt.xlabel('tv')
plt.ylabel('sales')
y_pred = lmodel1.predict(advdf.tv)
plt.plot(advdf.tv, y_pred, color='red')
plt.grid()
plt.show()
plt.close()
'''

## 선형회귀분석의 기본 충족 조건
# 잔차(실제값 - 예측값)항을 구하자
fitted = lmodel1.predict(advdf) # 예측값
residual = advdf['sales'] - fitted # 잔차
# print(f'실제값 : {advdf.sales[:5].values}') # [22.1 10.4  9.3 18.5 12.9]
# print(f'예측값 : {fitted[:5].values}') # [17.97077451  9.14797405  7.85022376 14.23439457 15.62721814]
# print(f'잔차 : {residual[:5].values}') # [ 4.12922549  1.25202595  1.44977624  4.26560543 -2.72721814]
# print(f'잔차 평균 : {np.mean(residual)}') # -1.4210854715202005e-15

# 1. 정규성 : 잔차가 정규성을 따르는지 확인
# - 모듈을 쓸 수 있고, 시각화할 수 있다.
from scipy.stats import shapiro
import statsmodels.api as sm # Quantile-Quantile plot을 지원한다.

stats, pv = shapiro(residual)
# print(f'shapiro-wilk test => 통계량 : {stats:.4f}\npvalue : {pv:.4f}')
# 통계량 : 0.9905
# pvalue : 0.2133 > 0.05 이므로 정규성 만족한다..
# print('정규성 만족' if pv > 0.05 else '정규성 위배 가능성')

# 시각화로 확인 : QQ plot
'''
sm.qqplot(residual, line='s')
plt.title('잔차 Q-Q plot')
plt.show()
plt.close() # 정규성 만족이나, 커브를 그려나가는 부분이 좋지 않다.
'''

# 2. 선형성 : 독립변수의 변화에 따라 종속변수도 일정 크기로 변화한다.
from statsmodels.stats.diagnostic import linear_reset # 선형성 확인, 모형 적합성 확인할 때 사용
reset_result = linear_reset(lmodel1, power=2, use_f=True)
# print(f'linear_reset test :\nF = {reset_result.fvalue:.4f}\npv = {reset_result.pvalue:.4f}')
# F = 3.7036
# pv = 0.0557
# print('선형성 만족' if reset_result.pvalue > 0.05 else '선형성 위배 가능성')

# 시각화로 확인 (잔차 확인)
'''
sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color':'red'}) # lowess 비모수적 추정
plt.plot([fitted.min(), fitted.max()], [0,0], '--', color='gray')
plt.show()
plt.close()
'''

# 3. 독립성 : 독립변수의 값이 서로 관련되지 않아야 한다.
# - 독립성 가정은 잔차 간에 자기상관이 없어야 한다.
# - 자기상관 : 회귀분석 등에서 잔차(오차)들이 서로 독립적이지 않고, 순서나 시간상으로 상관관계를 갖는 현상
# - Durbin-Watson(DW) 검정으로 확인 : 2에 근사하면 자기상관이 없다.


# 참고 : Cook's distance
# 하나의 관측치가 전체 모델에 얼마나 영향을 주는지 수치화한 자표
# outlier 관측에 활용 가능
from statsmodels.stats.outliers_influence import OLSInfluence
cd, _ = OLSInfluence(lmodel1).cooks_distance # 쿡의 거리값과 인덱스
# print(f'cook값 중 가장 큰 5개 관측치 확인:\n{cd.sort_values(ascending=False).head()}')
# 35     0.060494
# 178    0.056347
# 25     0.038873
# 175    0.037181
# 131    0.033895
# 이 값들이 outlier일 가능성이 있다.

# index 번째의 해당하는 원본 자료 확인
print(advdf.iloc[[35,178,25,175,131]])
#         tv  radio  newspaper  sales
# 35   290.7    4.1        8.5   12.8
# 178  276.7    2.3       23.7   11.8
# 25   262.9    3.5       19.5   12.0
# 175  276.9   48.9       41.8   27.0
# 131  265.2    2.9       43.0   12.7
# 설명/해석 : 대체적으로 tv 광고비는 높은데, 그에 반해 sales가 작다.
# model이 예측하기 어려운 포인트이다.
# 이 값들을 제외해서 검정도 해보고, 포함해서도 검정해본다.

# cook's distance 시각화
fig = sm.graphics.influence_plot(lmodel1, alpha=0.05, criterion='cooks')
# 오른쪽으로 갈 수록 이상치일 수 있다
plt.show()
plt.close() # 원이 클수록 좋지 않은 값이다.