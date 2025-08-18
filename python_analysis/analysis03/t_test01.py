'''
집단 차이분석 : 평균 또는 비율 차이를 분석

T-test와 ANOVA의 차이
- 두 집단 이하의 변수에 대한 평균 차이를 검정할 경우 T-test를 사용하여 검정통계량 T값을 구해 가설검정을 한다.
- 세 집단 이상의 변수에 대한 평균 차이를 검정할 경우에는 ANOVA를 이용하여 검정통계량 F값을 구해 가설검정을 한다.

# 핵심 아이디어
집단 평균차이(분자)와 집단 내 변동성(표준오차,표준편차 등 분모)을 비교하여, 
차이가 데이터의 불확실성(변동성)에 비해 얼마나 큰지를 계산한다.

- t 분포는 표분 평균을 이용해 정규분포의 평균을 해석할 떄 많이 사용한다.
- 대개의 경우 표본의 크기는 30개 이하일 때, t 분포를 따른다.
- t 검정은 '두 개 이하 집단의 평균의 차이가 우연에 의한 것인지 통계적으로 유의한 차이를 판단하는 통계적 절차이다.
'''
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 실습 1 - 어느 남성 집단의 평균 키 검정
# 귀무 : 집단의 평균 키가 177이다. (모집단 평균, 모수) bar x - mu = 0
# 대립 : 집단의 평균 키가 177이 아니다. bar x != mu
one_sample = [167.0, 182.7, 160.6, 176.8, 185.0]
# print(np.array(one_sample).mean()) # 174.42
# 177.0 과 174.42 평균의 차이가 있느냐?를 판단하는 과정이다.

result = stats.ttest_1samp(one_sample, popmean=177.0) # popmean : 모수 평균, 예상 평균
# print(f'statistic : {result[0]:.3f}\n' # -0.555
#       f'pvalue : {result[1]:.5f}' # 0.60847
#       )
# 결론 : p-value 0.60847 > 유의수준 0.05 이므로 귀무가설 채택.

# 시각화
sns.displot(one_sample, bins=4, kde=True, color='blue')
# plt.boxplot(one_sample)
plt.xlabel('data')
plt.ylabel('count')
# plt.show()
# plt.close()


# 실습 2
# 단일 모집단의 평균에 대한 가설검정
# A중학교 1학년 1반 학생들의 시험결과가 담긴 파일을 읽어처리 (국어 점수 평균 검정) - student.csv

# 귀무 : 학생들의 국어 점수의 평균은 80점이다.
# 대립 : 학생들의 국어 점수의 평균은 80점이 아니다.
data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv')
# print(data.head(2))
# data.info()
# print(data.describe()) # 요약통계량
#                국어          영어          수학
# count   20.000000   20.000000   20.000000
# mean    72.900000   71.750000   73.700000
# std     23.834738   19.931131   17.747053
# min     20.000000   30.000000   40.000000
# 25%     58.750000   63.750000   63.250000
# 50%     82.500000   70.000000   80.000000
# 75%     90.500000   87.000000   84.000000
# max    100.000000  100.000000  100.000000

# 정규성 검정 : one-sample t-test는 옵션이다.(해도 되고, 안해도 되고)
# 선형회귀선, 직선을 그렸을 때 값들이 선 근처에 있을 때는 정규성을 띄고 있다.
# print(f'정규성 검정 : {stats.shapiro(data.국어)}') # shapiro는 pvalue < 0.05 보다 작으므로 정규성을 만족하지 못함
# p-value = 0.01295975332132026
# 정규성 위배는 데이터 재가공, wilcoxon signed-rank test를 써야 더 안전하다.
# wilcoxon signed-rank test는 정규성을 가정하지 않는다.
from scipy.stats import wilcoxon
wilcoxon_res = wilcoxon(data.국어 - 80) # 평균 80과 비교
print(f'wilcox : {wilcoxon_res[1]:.5f}') # pvalue 0.39778 > 0.05 귀무가설 채택


res = stats.ttest_1samp(data.국어, popmean=80) # 얘만 쓰면 보고서가 빈약하기 때문에 wilcoxon 등 추가 활용해서 썰을 풀어준다.
# print(f'statistic : {res[0]:.3f}\n' # -1.332
#       f'pvalue : {res[1]:.5f}' # 0.19856
#       )

# 결론 : p-value 0.19856 > 유의수준 0.05 이므로 귀무가설 채택.

# 해석 : 정규성은 부족하지만, t-test와 wilcoxon은 같은 결론에 도달했다. (표본이 커지면 달라질 수 있음)
# 표본 수가 커지면 결과는 달라질 수 있다. 정규성 위배가 있어도 t-test 결과를 신뢰할 수 있다.

