import pandas as pd
import scipy.stats as stats
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
'''
[one-sample t 검정 : 문제1]
영사기에 사용되는 구형 백열전구의 수명은 250시간이라고 알려졌다. 
한국연구소에서 수명이 50시간 더 긴 새로운 백열전구를 개발하였다고 발표하였다. 
연구소의 발표결과가 맞는지 새로 개발된 백열전구를 임의로 수집하여 수명시간 관련 자료를 얻었다. 
한국연구소의 발표가 맞는지 새로운 백열전구의 수명을 분석하라.
   305 280 296 313 287 240 259 266 318 280 325 295 315 278
- 귀무가설 : 새로운 백열전구의 평균 수명은 300시간이다.
- 대립가설 : 새로운 백열전구의 평균 수명은 300시간이 아니다.
'''
# [one-sample t 검정 : 문제1]
# 가설
# 귀무 가설 : 새로운 백열전구의 수명은 300시간이다.
# 대립 가설 : 새로운 백열전구의 수명은 300시간이 아니다.
data1 = list(map(int,('305 280 296 313 287 240 259 266 318 280 325 295 315 278').split()))
# print(data1) 
# ['305', '280', '296', '313', '287', '240', '259', '266', '318', '280', '325', '295', '315', '278']
# print(f'평균 :{np.array(data1).mean():.4f}\n'
#       f'표준편차 :{np.array(data1).std():.4f}')
# 평균 :289.7857
# 표준편차 :23.6618

# print(f'p-value : {stats.shapiro(data1)[1]:.4f}') # 0.05 보다 크면 정규성 만족한다.
# 0.8209 > 0.05 이므로 정규성 만족한다.

result1 = stats.ttest_1samp(data1, popmean=300)
# print(f'statistic : {result1[0]:.3f}\n' # -1.556
#       f'pvalue : {result1[1]:.5f}' # 0.14361
#       )

# 결론 : p-value 0.14361 > 0.05 이므로 귀무가설을 채택한다.
# 우연히 발생된 결과이다. 새로운 백열전구 수명은 300시간이다.

# sns.histplot(data1, kde=True)
# plt.tight_layout()
# plt.show()
# plt.close()

# -----------------------------------
'''
[one-sample t 검정 : 문제2] 
국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간으로 파악되었다. 
A회사에서 생산된 노트북 평균시간과 차이가 있는지를 검정하기 위해서 A회사 노트북 150대를 랜덤하게 선정하여 검정을 실시한다.
실습 파일 : one_sample.csv
참고 : time에 공백을 제거할 땐 ***.time.replace("     ", "")
- 귀무 가설 : 국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간이다.
- 대립 가설 : 국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간이 아니다.
'''
# [one-sample t 검정 : 문제2]
# 가설
# 귀무 가설 : 국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간이다.
# 대립 가설 : 국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간이 아니다.
data2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv') # .astype({'time':'float64'})
# print(data2.columns) # ['no', '  gender', 'survey', 'time']
# data2.info()
data2['time'] = pd.to_numeric(data2['time'], errors = 'coerce')
tdata = pd.DataFrame(data2['time']).replace(' ',np.nan).dropna().reset_index(drop=True) # .fillna(means) : NaN 값 평균으로 채운다.
# tdata.info()
# print(tdata)

# print(f'평균 : {tdata['time'].mean():.4f}') # 평균 : 5.5569

# print(f'p-value : {stats.shapiro(tdata.time)[1]:.4f}') # 0.05 보다 크면 정규성 만족한다.
# 0.7242 > 0.05  이므로 정규성을 만족한다.

wilcoxon2 = wilcoxon(tdata.time - 5.2) # 평균 5.2과 비교
# print(f'wilcox : {wilcoxon2[1]:.5f}') # 0.00026 < 0.05 귀무가설 기각


result2 = stats.ttest_1samp(tdata['time'], popmean=5.2)
# print(result2)
# print(f'statistic : {result2[0]:.3f}\n' # 3.946
#       f'pvalue : {result2[1]:.5f}' # 0.00014
#       )
# 0.00014 < 0.05 이므로 귀무가설 기각한다.
# 분석된 데이터는 우연히 발생된 결과가 아니므로,
# 국내에서 생산된 대다수의 노트북 평균 사용 시간은 5.2 시간이 아니다.

# sns.histplot(tdata.time, kde=True)
# plt.tight_layout()
# plt.show()
# plt.close()

# -----------------------------------
'''
[one-sample t 검정 : 문제3] 
https://www.price.go.kr/tprice/portal/main/main.do 에서 
메뉴 중  가격동향 -> 개인서비스요금 -> 조회유형:지역별, 품목:미용 자료(엑셀)를 파일로 받아 미용 요금을 얻도록 하자. 
정부에서는 전국 평균 미용 요금이 15000원이라고 발표하였다. 이 발표가 맞는지 검정하시오.
'''
# [one-sample t 검정 : 문제3]
# 가설
# 귀무 가설 : 전국 평균 미용 요금이 15000원이다.
# 대립 가설 : 전국 평균 미용 요금이 15000원이 아니다.
data3 = pd.read_excel('python_analysis/analysis03/one_sample_ttest3.xlsx') # sheetname= 으로 원하는 시트에서 데이터를 가져올 수 있다.
# data3.info()
data3.replace(' ',np.nan).dropna()
# data3.info()
# print(data3.T.dropna())

ndata = data3.T.dropna()
ndata.drop(ndata.index[0:2], inplace=True)
# ndata.info()
ndata = pd.to_numeric(ndata.iloc[:,0],errors = 'coerce')
# print(ndata)
# ndata.info()

# print(f'평균 : {ndata.mean():.2f}') # 19512.88

# print(f'p-value : {stats.shapiro(ndata)[1]:.4f}') # 0.05 보다 크면 정규성 만족한다.
# 0.0581 > 0.05 이므로 정규성을 만족한다.

wilcoxon3 = wilcoxon(ndata - 15000) # 평균 15000과 비교
# print(f'wilcox : {wilcoxon3[1]:.5f}') # 0.00003 < 0.05 귀무가설 기각, 정규성을 위배한다.


result3 = stats.ttest_1samp(ndata, popmean=15000)
# print(result2)
# print(f'statistic : {result3[0]:.3f}\n' # 6.675
#       f'pvalue : {result3[1]:.5f}' # 0.00001
#       )

# 결론 : p-value 0.00001 < 0.05 이므로 귀무가설 기각한다.
# 데이터 결과는 우연히 도출된 것이 아니다. 6월달 전국 평균 미용 요금은 정부에서 공포한 평균 15000원 보다 증가했다.

sns.histplot(ndata, kde=True)
plt.show()
plt.close()