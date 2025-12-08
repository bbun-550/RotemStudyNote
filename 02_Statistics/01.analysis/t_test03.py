'''
비(눈) 여부(두 개의 집단)에 따른 음식점 매출액의 평균차이 검정
    1. 비가 올 때의 매출액
    2. 비가 오지 않을 때의 매출액
- 공통 칼럼이 연월일인 두 개의 파일을 조합을 해서 작업을 한다.
'''
# 귀무가설 : 강수량에 따른 음식점 매출액 평균차이는 없다.
# 대립가설 : 강수량에 따른 음식점 매출액 평균차이는 있다.

import numpy as np
import pandas as pd
import scipy.stats as stats

# 매출 자료 일기
sales_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tsales.csv', dtype={'YMD':'object'})
print(sales_data.head(2))
#         YMD    AMT  CNT
# 0  20190514      0    1
# 1  20190519  18000    1
sales_data.info() # 328 rows, 3 cols

# 날씨 자료 읽기
wt_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tweather.csv')
print(wt_data.head(2))
#    stnId          tm  avgTa  minTa  maxTa  sumRn  maxWs  avgWs  ddMes
# 0    108  2018-06-01   23.8   17.5   30.2    0.0    4.3    1.9    0.0
# 1    108  2018-06-02   23.4   17.6   30.1    0.0    4.5    2.0    0.0
wt_data.info() # 702 rows, 9cols

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

# print(data.isnull().sum()) # null 값 확인

# 강수 여부에 따른 매출액 평균 차이가 유의미한지 확인하자.
# 방법 1
# data['rain_yn'] = (data['sumRn'] > 0).astype(int) # 비가 오면 1, 안오면 0

# 방법 2
data['rain_yn'] = (data.loc[:,('sumRn')] > 0)*1
# print(data.head())

# AMT 매출액, rain_yn 강수 여부만 있으면 된다
sp = np.array(data.iloc[:, [1,4]])

tg1 = sp[sp[:,1]==0, 0] # 집단 1 : 비 안올 때의 매출액
tg2 = sp[sp[:,1]==1, 0] # 집단 2 : 비 올 때의 매출액
# print(f'tg1 : {tg1[:3]}\n') # [     0  50000 125000]
# print(f'tg2 : {tg2[:3]}\n') # [ 18000 274000 318000]

# 둘 평균의 차이를 boxplot으로 보자
import matplotlib.pyplot as plt
plt.boxplot([tg1, tg2], meanline=True, showmeans=True, notch=True) # meanline  notch : 중앙값

print(f'두 집단의 평균 : {np.mean(tg1):.4f} vs {np.mean(tg2):.4f}')
# 761040.2542 vs 757331.5217 이 둘이 차이가 있다고 봐야할지 말아야할지?

# plt.show()
# plt.close()

# 정규성 검정
print(len(tg1), ' ', len(tg2))
print(f'tg1 p-value : {stats.shapiro(tg1)[1]:.4f}\n') # 0.05 보다 크면 정규성 만족한다.
# tg1 p-value : 0.0561 > 0.05 정규성 만족
print(f'tg2 p-value : {stats.shapiro(tg2).pvalue:.4f}\n')
# tg2 p-value : 0.8828 > 0.05 정규성 만족

# 등분산성 검정 - levene
print(f'등분산성 : {stats.levene(tg1, tg2).pvalue:.4f}')
# 등분산성 : 0.7123 > 0.05 이므로 등분산성 만족

print(f'결과 pvalue : {stats.ttest_ind(tg1, tg2, equal_var=True)[1]:.4f}\n'
      f'결과 statistic : {stats.ttest_ind(tg1, tg2, equal_var=True)[0]:.4f}')
# statistic : 0.1011
# pvalue : 0.9195 > 0.05 이므로 귀무가설 채택 (실무에서는 귀무가설 기각이 많이 나온다.)
# 강수 여부에 따른 매출액 평균은 차이가 없다.