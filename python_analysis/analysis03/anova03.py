'''
## 온도에 따른 음식점 매출액의 평균 차이 검정
'''
# 가설
# 귀무가설 : 온도에 따른 음식점 매출액 평균 차이가 없다.
# 대립가설 : 온도에 따른 음식점 매출액 평균 차이가 있다.

import numpy as np
import pandas as pd
import scipy.stats as stats

# 매출 자료 일기
sales_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tsales.csv', dtype={'YMD':'object'})
# print(sales_data.head(2))
#         YMD    AMT  CNT
# 0  20190514      0    1
# 1  20190519  18000    1
# sales_data.info() # 328 rows, 3 cols

# 날씨 자료 읽기
wt_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tweather.csv')
# print(wt_data.head(2))
#    stnId          tm  avgTa  minTa  maxTa  sumRn  maxWs  avgWs  ddMes
# 0    108  2018-06-01   23.8   17.5   30.2    0.0    4.3    1.9    0.0
# 1    108  2018-06-02   23.4   17.6   30.1    0.0    4.5    2.0    0.0
# wt_data.info() # 702 rows, 9cols

# sales 자료를 기준으로 한다. 매출액 정보가 담겨 있기 때문이다.
# sales 데이터의 날씨를 기준으로 두 개의 자료를 병합 작업한다.
# 날짜 데이터의 생김새가 다르다. YMD - 20190514 , tm - 2018-06-01

wt_data.tm = wt_data.tm.map(lambda x:x.replace('-','')) # tm 날짜 형식을 YMD 형식과 동일하게 만들어준다
# print(wt_data.head(2)) # 성격은 같게 만들었지만 공통 칼럼이 없다. 공통 칼럼을 만들어준다

frame = sales_data.merge(wt_data, how='left', left_on='YMD', right_on='tm') # sales를 기준으로 merge했다.
# print(frame.head(2),'', len(frame)) # 328 rows

# print(frame.columns) # 보고 싶은 칼럼은 추출한다. YMD, AMT, maxTa, sumRn
# ['YMD', 'AMT', 'CNT', 'stnId', 'tm', 'avgTa', 'minTa', 'maxTa', 'sumRn', 'maxWs', 'avgWs', 'ddMes']

data = frame.iloc[:, [0,1,7,8]] # 날짜, 매출액, 최고온, 강수량 추출
# print(data.head(2))
#         YMD    AMT  maxTa  sumRn
# 0  20190514      0   26.9    0.0
# 1  20190519  18000   21.6   22.0

# -----여기부터 새로 입력

# 온도 구간 설정(3개)
# print(data.maxTa.describe())
# 최저 온도 -4.9 , 최고 온도 36.8

'''
일별 최고온도(연속형) 변수를 이용해 명목형(구간화 작업) 변수 추가한다.
'''
data['ta_gubun'] = pd.cut(data.maxTa, bins=[-5,8,24,37], labels=[0,1,2]) # 0 - 제일 추움
# print(data.head(2))
# print(data.ta_gubun.unique()) # [2, 1, 0] ; 카테고리 타입
# print(data.isnull().sum()) # null값 확인, 다행히 없음

# 최고온도를 세 그룹으로 나눈 뒤, 등분산, 정규성 검정 후 ANOVA 진행
x1 = np.array(data[data['ta_gubun']==0].AMT)
x2 = np.array(data[data['ta_gubun']==1].AMT)
x3 = np.array(data[data['ta_gubun']==2].AMT)

# print(f'{x1[:5]}, len : {len(x1)}')
# [1050500  770000 1054500  969000 1061500], len : 69

# 등분산 검정
# print(f'등분산성 levene : {stats.levene(x1, x2, x3).pvalue:.4f}') # pvalue: 0.0390
# 0.0390 < 0.05 등분산 만족하지 못했다.

# 정규성 검정
# print(f'x1 정규성 : {stats.shapiro(x1).pvalue:.6f}')
# print(f'x2 정규성 : {stats.shapiro(x2).pvalue:.6f}')
# print(f'x3 정규성 : {stats.shapiro(x3).pvalue:.6f}')
# x1 정규성 : 0.248192
# x2 정규성 : 0.038826
# x3 정규성 : 0.318299
# 정규성은 어느정도 만족이다. 
# 정규성 O, 등분산성X 이므로 welch 검정 염두하면 된다.

# 온도별 매출
spp = data.loc[:,['AMT','ta_gubun']]
# print(spp.groupby('ta_gubun').mean()) # 과학적 표기형식으로 출력된다.
## 피벗 테이블
# print(pd.pivot_table(spp, index=['ta_gubun'], aggfunc='mean')) # 같은 결과물 출력

# ANOVA 진행
# - group을 꺼낸다
sp = np.array(spp)
group1 = sp[sp[:,1]==0,0] # 모든 행의 1열
group2 = sp[sp[:,1]==1,0]
group3 = sp[sp[:,1]==2,0]

# print(stats.f_oneway(group1, group2, group3).pvalue)
# p-value 2.360737101089604e-34 < 0.05 이므로 귀무가설 기각한다. 온도는 매출과 관련이 있다.

'''
참고 : 등분산성을 만족하지 않으면 Welch's ANOVA를 사용한다.
- `pip install pingouin`
- 다양한 통계 테스트 지원: T-테스트, ANOVA, 상관 분석 등을 포함한다
- 효과 크기 계산: 다양한 효과 크기 지표를 제공한다
- 모델링 도구: 선형 회귀, ANCOVA, 중복 분석 등을 지원한다
- 데이터 시각화: 통계 결과를 이해하기 쉬운 그래프로 표현할 수 있다
- 사용자 친화적인 API: Pandas DataFrame과 함께 사용하기 쉽도록 설계됐다
'''
from pingouin import welch_anova
print(f'welch :\n{welch_anova(dv='AMT', between='ta_gubun', data=data)}')
#      Source  ddof1     ddof2           F         p-unc       np2
# 0  ta_gubun      2  189.6514  122.221242  7.907874e-35  0.379038
# p-value : 7.907874e-35 < 0.05 이므로 귀무기각이다.

# 참고 : 정규성을 만족하지 않으면 Kruskal-Wallis test를 사용한다.
print(f'Kruskal-Wallis : {stats.kruskal(group1, group2, group3).pvalue}')
# p-value : 1.5278142583114522e-29 < 0.05 귀무기간
# 온도에 따라 매출액의 차이가 유의미하다.

# 둘 다 만족하지 않을 경우 데이터 재가공.

# 사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
turkyResult = pairwise_tukeyhsd(endog=spp.AMT, groups=spp.ta_gubun)
# print(turkyResult)
#        Multiple Comparison of Means - Tukey HSD, FWER=0.05       
# =================================================================
# group1 group2   meandiff   p-adj    lower        upper     reject
# -----------------------------------------------------------------
#      0      1 -214255.4486   0.0  -296755.647 -131755.2503   True
#      0      2 -478651.3813   0.0 -561484.4539 -395818.3088   True
#      1      2 -264395.9327   0.0 -333326.1167 -195465.7488   True
# -----------------------------------------------------------------

# 시각화
turkyResult.plot_simultaneous(xlabel='mean',ylabel='group')
plt.show()
plt.close()
# 겹치는 부분이 없다.
# ![[anova03_사후검정시각화.png]]