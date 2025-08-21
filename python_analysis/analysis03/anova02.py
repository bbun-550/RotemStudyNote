'''
## 일원분산분석 연습
- 강남구에 있는 GS 편의점 3개 지역 알바생의 급여에 대한 평균의 차이를 검정해보자.
'''
# 가설
# 귀무가설 : GS 편의점 3개 지역 알바생의 급여에 대한 평균의 차이가 없다.
# 대립가설 : GS 편의점 3개 지역 알바생의 급여에 대한 평균의 차이가 있다.
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3.txt"
# pandas로 불러오기
# data = pd.read_csv(url, header=None)
# print(data.values)

# numpy로 불러오기
data = np.genfromtxt(url, delimiter=',') # delimiter : 구분자
# print(f"{data}\n{type(data)}\n{data.shape}") # 리스트로 출력한다. type은 ndarray이다. (22, 2)

# 3개의 집단 월급, 평균 얻기
gr1 = data[data[:,1]==1, 0] # 1번 지역의 월급
gr2 = data[data[:,1]==2, 0] # 1번 지역의 월급
gr3 = data[data[:,1]==3, 0] # 1번 지역의 월급

# print(f'gr1 평균: {np.mean(gr1):.4f}\n' 
#       f'gr2 평균: {np.mean(gr2):.4f}\n'
#       f'gr3 평균: {np.mean(gr3):.4f}')
# gr1 평균: 316.6250
# gr2 평균: 256.4444
# gr3 평균: 278.0000

# 정규성 검정
# print(f'gr1 정규성 : {stats.shapiro(gr1).pvalue:.6f}') # pvalue: 0.333683
# print(f'gr2 정규성 : {stats.shapiro(gr2).pvalue:.6f}') # pvalue: 0.656105
# print(f'gr3 정규성 : {stats.shapiro(gr3).pvalue:.6f}') # pvalue: 0.832481
# 셋 다 정규성 만족한다.


# 등분산성 검정
# print(f'등분산성 levene : {stats.levene(gr1, gr2, gr3).pvalue:.4f}') # pvalue: 0.0458
# print(f'등분산성 bartlett : {stats.bartlett(gr1, gr2, gr3).pvalue:.4f}') # pvalue: 0.3508
# levene 0.05보다 작지만 근사하기 때문에 만족이라고 본다.

# 데이터 분포 확인해보자
# plt.boxplot([gr1,gr2,gr3], showmeans=True)
# plt.show()
# plt.close()
# 차이는 있어 보이는데 진짜 평균의 차이가 있어?

'''
## ANOVA 검정 방법1
- 수식 : anova_lm
- (1분 이하로 설명할 줄 알아야 한다.)
'''
# ndarray로 되어 있기 때문에 바꿔줘야 한다.
# print(type(data)) # <class 'numpy.ndarray'>
df = pd.DataFrame(data, columns=['pay','group'])
# print(df)

lmodel = ols('pay ~ C(group)', data=df).fit() # C 연산자를 꼭 써줘야한다(범주형일 때). ols 잔차제곱합을 최소화 해준다
# .fit() : 학습시켜준다.

# print(anova_lm(lmodel, typ=1))
#             df        sum_sq      mean_sq         F    PR(>F)
# C(group)   2.0  15515.766414  7757.883207  3.711336  0.043589
# Residual  19.0  39716.097222  2090.320906       NaN       NaN

# 결과 : pvalue 0.043589 < 0.05 귀무가설 기각

# 복습 : f-value = MSR/MSE
## 추가 : MSR = SSR / df


'''
## ANOVA 검정 방법2
- f_oneway
- 분산분석표가 출력되지 않고, 그냥 값만 반환한다.
'''
f_statistic, pvalue = stats.f_oneway(gr1, gr2, gr3)
# print(f'f_statistic : {f_statistic:.4f}\n'
#       f'pvalue : {pvalue:.4f}')
# f_statistic : 3.7113 = BV/WV or 집단 간 분산/집단 내 분산
# pvalue : 0.0436 < 0.05 귀무가설 ; 위 방법과 동일한 값 도출 됐다.
# GS 편의점 3개 지역 알바생의 급여에 대한 평균의 차이가 있다. 
# 그럼 얼마나 차이가 나?

# 사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkyResult = pairwise_tukeyhsd(endog=df.pay, groups=df.group)
# print(turkyResult)
#  Multiple Comparison of Means - Tukey HSD, FWER=0.05  
# ======================================================
# group1 group2 meandiff p-adj    lower    upper  reject
# ------------------------------------------------------
#    1.0    2.0 -60.1806 0.0355  -116.619 -3.7421   True # 차이 있다!
#    1.0    3.0  -38.625 0.3215 -104.8404 27.5904  False
#    2.0    3.0  21.5556 0.6802  -43.2295 86.3406  False
# ------------------------------------------------------

# 시각화
# turkyResult.plot_simultaneous(xlabel='mean',ylabel='group')
# plt.show()
# plt.close()
# 1번과 2번 겹치는 부분이 없다. 즉, 1과 2 간 평균 차이가 있다.