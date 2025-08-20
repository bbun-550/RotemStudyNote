'''
## 세 개 이상의 모집단에 대한 가설검정 – 분산분석
- ‘분산분석’이라는 용어는 분산이 발생한 과정을 분석하여 
    요인에 의한 분산과 요인을 통해 나누어진 각 집단 내의 분산으로 나누고 요인에 의한 분산이 의미있는 크기를 가지는지를 검정하는 것을 의미한다.
- 세 집단 이상의 평균 비교에서는 독립인 두 집단의 평균 비교를 반복하여 실시할 경우에 제1종오류가 증가하게 되어 문제가 발생한다.
- 이를 해결하기 위해 Fisher가 개발한 분산분석(ANOVA, ANalysis Of Variance)을 이용하게 된다.

- 분산의 성질과 원리를 이용하여, 평균의 차이를 분석한다.
- 즉, 평균을 직접 비교하지 않고 집단 내 분산과 집단 간 분산을 이용하여 집단의 평균이 서로 다른지 확인하는 방법이다.
- f-value(비율 값) = 그룹 간 분산(BV) / 그룹 내 분산(WV)

#### 실습1 - 세 가지 교육방법을 적용하여 1개월 동안 교육받은 교육생 80명을 대상으로 실기시험을 실시
- 교육방법이 교육생 실기시험에 영향을 주고 있다.
- 교육방법(세가지 방법)이 독립변수, 실기시험이 종속변수 -> 일원분산분석 one-way ANOVA
    - 일원분산분석 : 복수의 집단을 대상으로 집단을 구분하는 요인이 하나일 때 사용하는 방법이다.
- three_sample.csv
'''
# 가설
# 귀무가설 : 세 가지 교육방법을 통한 교육생 실기시험 평균에 차이가 없다.
# 대립가설 : 세 가지 교육방법을 통한 교육생 실기시험 평균에 차이가 있다.

import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols # 최소제곱으로 기울기와 절편을 구할 수 있다(직선을 구할 수 있다는 의미이며, 회귀분석에서 중요하다)
# statsmodels : 시계열, 회귀분석, 분류모델 작성...등에 필요한 라이브러리를 갖고 있다.

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/three_sample.csv')
# 필요한 칼럼은 method랑 score이다.
# print(data.head(2))
# print(data.columns) # ['no', 'method', 'survey', 'score']
# print(data.shape) # (80, 4)
# print(data['method'].unique()) # [1 3 2]
# print(data.describe()) # 결측치, 평균, 이상치 ... 확인 -> score max에 이상치 발견!

# outlier 차트로 확인하자.
import matplotlib.pyplot as plt

# plt.hist(data['score']) # 이상치가 있다.
# plt.boxplot(data['score']) # 사실 이상치 2개였다.
# plt.show()
# plt.close()

# 이상치 제거(방법은 다양하다)
data = data.query('score <= 100')
# print(len(data)) # 78 ; 원래 80이었는데 이상치 제거 후 78로 줄어들었다.

# 필요한 데이터는 method랑 score이다. result DataFrame 만들어준다.
result = data[['method','score']]
# print(result)

# 교육방법 method 분리
m1 = result[result['method']==1] # 교육 방법1
m2 = result[result['method']==2] # 교육 방법2
m3 = result[result['method']==3] # 교육 방법3

score1 = m1['score']
score2 = m2['score']
score3 = m3['score']


# 정규성 검정
# print(f'score1 정규성 : {stats.shapiro(score1).pvalue:.6f}') # pvalue: 0.174674
# print(f'score2 정규성 : {stats.shapiro(score2).pvalue:.6f}') # pvalue: 0.331900
# print(f'score3 정규성 : {stats.shapiro(score3).pvalue:.6f}') # pvalue: 0.115586
# 교육방법 셋 다 p-value > 0.05 이므로 정규성을 만족한다.
## 정규성 만족하지 않는다면 ANOVA를 쓸 수 없는건가? 그럼 어떤 검정을 써야하지?
## Kruskal-Wallis test 방법 쓴다.

## 정규성 2개씩 확인하는 방법
# print(stats.ks_2samp(score1, score2)) # 두 집단의 동일 분포 여부 확인용 

# 등분산성(복수 집단 분산의 치우침 정도 확인) 3가지 있음
# print(f'등분산성 levene : {stats.levene(score1, score2, score3).pvalue:.4f}') # pvalue: 0.1132
# print(f'등분산성 fligner : {stats.fligner(score1, score2, score3).pvalue:.4f}') # pvalue: 0.1085
# print(f'등분산성 bartlett : {stats.bartlett(score1, score2, score3).pvalue:.4f}') # (비모수) pvalue: 0.1055
# 교육방법 셋 다 p-value > 0.05 이므로 등분산성을 만족한다.
## 정규성은 만족하고 등분산성 불만족시 Welch’s ANOVA 사용한다.

# 교차표 등 작성 할 수 있다. but pass

# ANOVA 검정
import statsmodels.api as sm
## 회귀분석 이야기 : 최소제곱 ... 나중에 배울 예정이다
# 단일 선형회귀 모델 만드는 과정 ; reg가 선형회귀모델이다.
reg = ols("data['score'] ~ C(data['method'])", data=data).fit() # 종속변수 적고 틸드 ~, C연산자는 범주형일 때 사용한다,

# 분산 분석표를 이용해서 분산결과 작성했다.
table = sm.stats.anova_lm(reg, type=2)
# lm : linear regression model, type 기본값이 2(1,2,3) ; type별 차이점?

# print(table) 
# 결과 : pvalue = 0.939639 > 0.05 이므로 귀무 채택
# 세 가지 교육방법을 통한 교육생 실기시험 평균에 차이가 없다. 수집된 데이터는 우연히 발생됐다.

#                        df(자유도) sum_sq(제곱합)  mean_sq(제곱평균)  F                  PR(>F)
# C(data['method'])회귀   2.0     28.907967(SSR)   14.453984(MSR)  0.062312(검정통계량)  0.939639
# Residual 잔차           75.0  17397.207418(SSE)  231.962766(MSE)     NaN             NaN

# f-value로 p-value를 구했다.

# 참고 : residual는 잔차. 데이터를 관통하는 추세선을 그릴 수 있다. 이때 추세선(=회귀선 y=wx+b)을 잔차로 구한다.
## SSR : 추세선과 데이터 간의 차이
## SSE 잔차에러 : 평균과 데이터 간의 차이

## SSR/MSE = F-value -> p-value

# 사후검정 post hoc test
# - 분산분석은 집단의 평균에 차이 여부만 알려줄 뿐, 집단 간의 평균 차이는 알려주지 않는다.
# - (즉, 분산분석 ANOVA는 차이가 있다/없다만 알려준다)
# - 각 집단 간의 평균 차이를 확인하기 위해 사후검정 실시!
from statsmodels.stats.multicomp import pairwise_tukeyhsd # multicomp ; 여러개 비교
turResult = pairwise_tukeyhsd(endog=data.score, groups=data.method)
# print(turResult)
# Multiple Comparison of Means - Tukey HSD, FWER=0.05 
# ====================================================
# group1 group2 meandiff p-adj   lower   upper  reject
# ----------------------------------------------------
#      1      2   0.9725 0.9702 -8.9458 10.8909  False # 1과 2 비교 /.../차이가 없으면 False, 있으면 True 출력
#      1      3   1.4904 0.9363 -8.8183  11.799  False
#      2      3   0.5179 0.9918 -9.6125 10.6483  False
# ----------------------------------------------------

# 결론 : 세 가지 교육방법에 차이가 없다.

# 사후검정 결과물 시각화
turResult.plot_simultaneous(xlabel='mean', ylabel='group')
# plt.show()
# plt.close()
# 겹치는 부분이 없을 수록 차이가 많다
# but 겹치는 부분이 많다.
# ![[사후검정시각화_turResult.png]]