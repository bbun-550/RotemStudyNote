'''
독립 표본 검정
- 두 집단의 평균의 차이 검정
- 서로 다른 두 집단의 평균에 대한 통계 검정에 주로 사용한다.
- 비교를 위해 평균(분자)과 표준편차(분모) 통계량을 사용한다.
- 평균값의 차이가 얼마인지, 표준편차는 얼마나 다른지 확인해서 
    분석 대상인 두 자료가 같을 가능성의 우연의 범위 5%에 들어가는지 판별한다.

- 결국 t-test는 두 집단의 평균과 표준편차 *비율*에 대한 대조 검정법이다.

* 서로 독립인 두집단의 평균 차이 검정 (independentsamplest-test)
남녀의 성적, A반과 B반의 키, 경기도와 충청도의 소득 따위의 서로 독립인 두 집단에서 얻은 표본을 독립 표본(twosample)이라고 한다.
- 실습1)남녀 두 집단 간 파이썬 시험의 평균 차이 검정

# 가설
# 귀무 가설 : 남녀 두 집단 간 피이썬 시험의 평균 차이가 없다
# 대립 가설 : 남녀 두 집단 간 피이썬 시험의 평균 차이가 있다
# 95% 신뢰수준에서 우연히 발생할 확률이 5%보다 작은지 검정한다. (작다면 귀무 기각)
# 선행 조건 : 두 집단 자료는 정규분포를 따른다.
    - 분산이 동일하다(등분산성)
'''
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import wilcoxon

male = [75, 85, 100, 72.5, 86.5]
female = [63.2, 76, 52, 100, 70]
# print(f'male : {np.mean(male)}\n'
#       f'female : {np.mean(female)}')
# male : 83.8
# female : 72.24
# 둘 평균의 차이가 있다??

two_sample = stats.ttest_ind(male, female) 
two_sample = stats.ttest_ind(male, female, equal_var=True) # equal var = True가 기본값 ; 등분산성이 같다는 가정하에 작업
print(f'statistic : {two_sample[0]:.4f}\n'
      f'p-value : {two_sample[1]:.4f}\n')
# statistic : 1.2332
# p-value : 0.2525


# 해석 : pvalue 0.2525 > 0.05 이므로 귀무 채택이다.
# 두 집단의 평균 차이는 없다. 통계적으로 유의하지 않다.
# 남녀 두 집단 간 파이썬 시험의 평균 차이는 없다.두 집단 평균 차이가 유의하지 않다

# sample 수가 적으면 pvalue가 높게 나올 수도 있다. 검정력을 증가시키려면 표본 수를 늘리거나, 신뢰수준을 다른 값을 쓰는 것도 방법이다.

# -------------------------

'''
등분산 검정
- bartlett : scipy.stats.bartlett
- fligner : scipy.stats.fligner
- levene : scipy.stats.levene
'''
from scipy.stats import levene
leven_stat, leven_p, = levene(male, female)
print(f'leven 통계량 : {leven_stat:.4f}\n'
      f'p-value : {leven_p:.4f}')
# leven 통계량 : 0.5095
# p-value : 0.4957

if leven_p > 0.05:
    print('등분산성을 가정한다. (분산이 같다고 할 수 있다)')
else:
    print('분산이 같다고 할 수 없다. 등분산 가정이 부적절하다.')

'''
만약 등분산성 가정이 부적절한 경우,
    Welch's t-test 사용을 권장한다.
'''
# welch test 할 필요 없지만 해보기로 한다.

welch_result= stats.ttest_ind(male, female, equal_var=False) # equal_var=False 분산이 다를 수 있음을 가정하고 계산한다.
print(f'statistic : {welch_result[0]:.4f}\n'
      f'p-value : {welch_result[1]:.4f}\n')
# statistic : 1.2332
# p-value : 0.2595 # 위에 등분산성을 가정했을 때 값 p-value : 0.2525