'''
교차분석(카이제곱) 가설검정
Cross-Tabulation analysis
정식명칭 : Pearson's chi-square test <-- 두 불연속변수(범주형) 간의 상관관계를 측정하는 기법
- 데이터를 파악할 때 중심위치(평균)과 퍼짐정도(분산 ; 표준편차보다 극단적)가 중요한데
    카이제곱 검정은 표준편차 제곱에 대한 분포이다.(분산에 대한 분포)
- 범주형 자료를 대상으로 교차 빈도에 대한 검정통계량(표본 데이터), 유의성을 검증해 주는 추론통계 기법

2가지 유형
- 일원카이제곱 (변인 단수) : 적합도, 선호도 검증, 교차 분할표 사용 X
- 이원카이제곱 (변인 복수) : 독립성, 동질성 검증, 교차 분할표 사용 O

유의확률 p-value에 의해 집단 간에 차이 여부를 가설로 검증

# 교차분석 흐름 이해용 실습 : 수식((관측값-기대값)^2/기대값의 합)으로 카이제곱 구하기, 함수로 구하기
'''
import pandas as pd
data = pd.read_csv('./pass_cross.csv',)
# print(data.head())

# 귀무가설 H0 : 벼락치기와 합격 여부는 관계가 없다
# 대립가설 H1 : 벼락치기와 합격 여부는 관계가 있다

# print(data[(data['공부함'] == 1) & (data['합격'] == 1)].shape[0]) # 18명
# print(data[(data['공부함'] == 1) & (data['불합격'] == 1)].shape[0]) # 7명

# 빈도표 작성
ctab = pd.crosstab(index=data['공부안함'], columns=data['불합격'], margins=True) # colnames vs columns
ctab.columns = ['합격','불합격','행합']
ctab.index = ['공부함','공부안함','열합']
print(ctab)

# 방법 1 : 수식 사용
# 기대도수 Expected frequency? (각 행의 주변합) + (각 열의 주변합) / 총합
# chi2 = (18 - 15) ** 2 / 15 + ...
# chi2 = 3.0
# 임계값? 카이제곱 표 사용하면 됨
# 자유도 degree of freedom? (행의 개수 - 1) * (열의개수 - 1) = 1
# Critical Value 임계값 = 3.84
# 결론 : 카이제곱 검정 통계량 3.0은 cv 3.84 보다 작으므로 귀무채택역에 있다.
# 그러므로 대립가설을 기각한다. 즉, 벼락치기와 합격 여부는 관계가 없다.


# 방법 2 : 함수 사용
# p-value 유의확률 사용
import scipy.stats as stats
chi2, pvalue, dof, expected = stats.chi2_contingency(ctab)
# print(chi2, pvalue, dof, expected)
# chi2 = 3.0 
# pvalue = 0.5578254003710748 
# dof = 4 
# expected = 
# [[15. 10. 25.]
#  [15. 10. 25.]
#  [30. 20. 50.]]

msg = "Test statisic : {}, p-value : {:.6f}"
# print(msg.format(chi2, pvalue)) # Test statisic : 3.0, p-value : 0.557825
# 결론 : 유의확률 p-value(0.557825) > 유의수준 α(0.05) 이므로 귀무가설 채택
# 새로운 주장을 위해 수집된 data는 우연히 발생한 자료라고 할 수 있다.