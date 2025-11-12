'''
일원 카이제곱 검정 : 변인이 1개
적합도, 선호도 검정
- 실험을 통해 얻은 여러 관찰값이 어떤 이론적 분포를 따르고 있는지 확인하는 검정
- eg. 꽃 색깔의 표현 분리 비율이 3:1이 맞는가?

<적합도 검정 실습>
주사위를 60회 던져서 나온 관측도수 / 기대도수가 아래와 같이 나온 경우에 
이 주사위는 적합한 주사위가 맞는가를 일원카이제곱 검정으로 분석하자.
'''
# 주사위눈금  1 2  3  4 5 6 
# 관측도수    4 6 17 16 8 9 
# 기대도수  10 10 10 10 10 10 

# 가설
# 귀무가설 H0 : 기대치와 관찰치는 차이가 없다.(현재 주사위는 게임에 적합하다)
# 대립가설 H1 : 기대치와 관찰치는 차이가 있다.(현재 주사위는 게임에 적합하지 않다)

import pandas as pd
import scipy.stats as stats

data = [4, 6, 17, 16, 8, 9] # 관측값
exp = [10, 10, 10, 10, 10, 10] # 기대값

# print(stats.chisquare(data))
# Power_divergenceResult(statistic=np.float64(14.200000000000001), pvalue=np.float64(0.014387678176921308))
# X2 = 14.20 , p-value = 0.01438
# 결론 : p-value < α(0.05) 이므로 귀무 기각
# 이 주사위는 게임에 적합하지 않다.
# 관측값은 우연히 발생한 것이 아니라 어떠한 원인에 의해 얻어진 값이다.

# print(stats.chisquare(data, exp)) # 관찰빈도, 기대빈도(생략가능). 결과는 위랑 똑같음

result = stats.chisquare(data, exp) # 출력 방식
print(f'chic2 : {result[0]:.2f},\np-value : {result[1]:.6f}')

# ----------------------

'''
<선호도 분석 실습>
- 5개의 스포츠 음료에 대한 선호도에 차이가 있는지 검정하기
'''

sdata = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinkdata.csv')
# print(sdata)
#   음료종류  관측도수
# 0   s1    41
# 1   s2    30
# 2   s3    51
# 3   s4    71
# 4   s5    61

# print(stats.chisquare(sdata['관측도수']))
# # Power_divergenceResult(statistic=np.float64(20.488188976377952), pvalue=np.float64(0.00039991784008227264))

# print(f'chic2 : {stats.chisquare(sdata['관측도수'])[0]:.2f}\np-value : {stats.chisquare(sdata['관측도수'])[1]:.6f}')
# chic2 : 20.49
# p-value : 0.000400

# 결과 : p-value(0.000400) < α(0.05) 이므로 귀무 기각, 대립 채택.
# 음료에 대한 선호도 차이가 있다.

# 선호도 결과 시각화
# 어떤 음료가 기대보다 많이 선택했는지
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='applegothic') # malgun gothic'
plt.rcParams['axes.unicode_minus'] = False

# 기대도수 계산
total = sdata['관측도수'].sum()
# expected = sdata['관측도수'].mean()
expected = [total / len(sdata)] * len(sdata)
print(f'expected : {expected}') # 50.8

x = np.arange(len(sdata))
width = 0.35 # 막대 너비

plt.figure(figsize=(9,5))
plt.bar(x-width/2, sdata['관측도수'], width=width, label='관측도수')
plt.bar(x-width/2, expected, width=width, label='기대도수', alpha=0.6)
plt.xticks(x, sdata['음료종류'])
plt.xlabel('음료종류')
plt.ylabel('도수')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.show()
plt.close()

# 그래프와 카이제곱 검정 결과를 바탕으로 어떤 음료가 더 인기 있는지 구체적으로 분석
# 총합과 기대도수 이미 구함. 밋밋한 DataFrame에 데이터 추가.
# 차이 계산
sdata['기대도수'] = expected
sdata['차이(관측-기대)'] = sdata['관측도수'] - sdata['기대도수']
sdata['차이비율(%)'] = round(sdata['차이(관측-기대)'] / expected * 100 , 2)
# print(sdata.head(3))
#   음료종류  관측도수  기대도수  차이(관측-기대)  차이비율(%)
# 0   s1    41        50.8       -9.8       -19.29
# 1   s2    30        50.8      -20.8       -40.94
# 2   s3    51        50.8        0.2         0.39

sdata.sort_values(by='차이(관측-기대)', ascending=False, inplace=True) # by : 기준이 누구야, 내림차순
sdata.reset_index(drop=True, inplace=True) # 설정 인덱스를 제거하고 기본 인덱스(0,1,2, ... , n)으로 변경하는 메서드
# print(sdata.head(3))
#   음료종류  관측도수  기대도수  차이(관측-기대)  차이비율(%)
# 0   s4    71       50.8       20.2          39.76
# 1   s5    61       50.8       10.2          20.08
# 2   s3    51       50.8        0.2           0.39