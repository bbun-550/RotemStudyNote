'''
## 추론통계 분석 중 비율검정 - 비율검정 특징
- 집단의 비율이 어떤 특정한 값과 같은지를 검증.
- 비율 차이 검정통계량을 바탕으로 귀무가설의 기각 여부를 결정.

# one-sample
- A회사에는 100명 중에 45명이 흡연을 한다.국가통계를 보니 국민흡연율은 35%라고 한다. 비율이같냐?
- 귀무 : A회사 직원들의 흡연율과 국민 흡연율의 비율이 같다.
- 대립 : A회사 직원들의 흡연율과 국민 흡연율의 비율이 같지 않다.
'''
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
count = np.array([45]) # 45명
nobs = np.array([100]) # 전체 100명
val = 0.35 # 흡연율

z, pvalue = proportions_ztest(count=count, nobs=nobs, value=val)
print(z) # [2.01007563]

print(pvalue) # [0.04442318] < 0.05 이므로 귀무 기각이다. 비율이 다르다.

'''
# two-sample
- A회사 사람들 300명 중 100명이 커피를 마시고, B회사 사람들 400명 중 170명이 커피를 마셨다. 비율이같냐?
- 귀무 : A회사와 B회사의 커피 마시는 사람의 비율이 같다.
- 대립 : A회사와 B회사의 커피 마시는 사람의 비율이 같지 않다.
'''
count = np.array([100, 170]) # A, B회사 커피 마시는 사람
nobs = np.array([300, 400])
z, pvalue = proportions_ztest(count=count, nobs=nobs, value=val)
print(z)

print(pvalue) # 1.5028294294082938e-32 < 0.05 이므로 귀무기각이다. 비율이 다르다.

# -----------------------------------
'''
## 이항 검정
- 결과가 두 가지 값을 가지는 확률변수의 분포를 판단 하는데 효과적이다.
- 예) 10명의 자경증 시험 합격자 중 여성이 6명이었다고 할 때, 
    '여성이 남성보다 합격률이 높다고 할 수 있는가?'
- one_sample.csv
'''
import pandas as pd
import scipy.stats as stats

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv')
print(data.head(3))

# 교차표
ctab = pd.crosstab(index=data['survey'], columns='count')
ctab.index = ['불만족', '만족']
print(ctab)
# 불만족       14
# 만족       136

# 귀무가설 : 직원 대상으로 고객 대응 교육 후 고객 안내 서비스 만족률이 80% 이다. 
# 대립가설 : 직원 대상으로 고객 대응 교육 후 고객 안내 서비스 만족률이 80% 아니다.

# 양측검정 : 방향성이 없다.
result = stats.binomtest(k=136, n=150,p=0.8,alternative="two-sided")
print(result.pvalue) # 0.0006734701362867024 < 0.05 이므로 귀무가설 기각. 
'''
binomtest()
- k: 관측된 성공 횟수 (정수)
- n: 전체 시행 횟수 (정수)
- p: 귀무 가설 하의 성공 확률 (0과 1 사이의 부동 소수점)
- alternative: 대립 가설 설정 ('two-sided', 'greater', 'less')
'''

# 단측검정 : 방향성이 있다. (80%보다 크다라고 가정하고 검증한다.)
result = stats.binomtest(k=136, n=150,p=0.8,alternative="greater") 
print(result.pvalue) # 0.00031794019219854805 < 0.05 이므로 귀무가설 기각. 80%보다 크지 않다.