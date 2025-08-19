import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import levene
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib.pyplot as plt

'''
[two-sample t 검정 : 문제1] 
다음 데이터는 동일한 상품의 포장지 색상에 따른 매출액에 대한 자료이다. 
포장지 색상에 따른 제품의 매출액에 차이가 존재하는지 검정하시오.
   blue : 70 68 82 78 72 68 67 68 88 60 80
   red : 60 65 55 58 67 59 61 68 77 66 66
'''

'''
# 귀무가설 : 포장지 색상에 따른 제품의 매출액에 차이가 없다
# 대립가설 : 포장지 색상에 따른 제품의 매출액에 차이가 있다

blue = list(map(int,('70 68 82 78 72 68 67 68 88 60 80').split()))
red = list(map(int,('60 65 55 58 67 59 61 68 77 66 66').split()))
print(blue)
print(red)

print(f'두 집단의 평균 : {np.mean(blue):.4f} vs {np.mean(red):.4f}')
# 두 집단의 평균 : 72.8182 vs 63.8182

# 정규성 검사 - shapiro
print(f'blue p-value : {stats.shapiro(blue).pvalue:.4f}\n') # 0.05 보다 크면 정규성 만족한다.
# blue p-value : 0.5102 > 0.05 정규성 만족

print(f'red p-value : {stats.shapiro(red).pvalue:.4f}\n')
# red p-value : 0.5348 > 0.05 정규성 만족

# 등분산성 검정 - levene
print(f'등분산성 : {stats.levene(blue, red).pvalue:.4f}') # 0.05 보다 크면 등분산성 만족한다.
# 등분산성 : 0.4392 > 0.05 등분산성 만족

print(f'결과 statistic : {stats.ttest_ind(blue, red, equal_var=True)[0]:.4f}\n'
      f'결과 pvalue : {stats.ttest_ind(blue, red, equal_var=True)[1]:.4f}')
# 결과 statistic : 2.9280
# 결과 pvalue : 0.0083 < 0.05 이므로 귀무가설 기각한다.
# 포장지 색상에 따른 제품의 매출액 차이가 있다.
'''

'''
[two-sample t 검정 : 문제2]  
아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여 혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.
  남자 : 0.9 2.2 1.6 2.8 4.2 3.7 2.6 2.9 3.3 1.2 3.2 2.7 3.8 4.5 4 2.2 0.8 0.5 0.3 5.3 5.7 2.3 9.8
  여자 : 1.4 2.7 2.1 1.8 3.3 3.2 1.6 1.9 2.3 2.5 2.3 1.4 2.6 3.5 2.1 6.6 7.7 8.8 6.6 6.4
'''
'''
# 귀무가설 : 성별에 따른 혈관 내의 콜레스테롤 양에 차이가 없다.
# 대립가설 : 성별에 따른 혈관 내의 콜레스테롤 양에 차이가 있다.

import random
male = list(map(float,('0.9 2.2 1.6 2.8 4.2 3.7 2.6 2.9 3.3 1.2 3.2 2.7 3.8 4.5 4 2.2 0.8 0.5 0.3 5.3 5.7 2.3 9.8').split()))
female = list(map(float,('1.4 2.7 2.1 1.8 3.3 3.2 1.6 1.9 2.3 2.5 2.3 1.4 2.6 3.5 2.1 6.6 7.7 8.8 6.6 6.4').split()))
# print(male)
# print(set(female), len(female), len(set(female)))
random.seed(23)
r_male = random.sample(male, 15)
r_female = random.sample(female, 15)
# print(set(r_male), len(r_male), len(set(r_male)))
# print(set(r_female), len(r_female), len(set(r_female)))


print(f'두 집단의 평균 : {np.mean(r_male):.4f} vs {np.mean(r_female):.4f}')
# 두 집단의 평균 : 3.0467 vs 3.2267

# 정규성 검사 - shapiro
print(f'male p-value : {stats.shapiro(r_male).pvalue:.4f}\n') # 0.05 보다 크면 정규성 만족한다.
# male p-value : 0.0272 < 0.05 정규성 위배

print(f'female p-value : {stats.shapiro(r_female).pvalue:.4f}\n')
# female p-value : 0.0013 < 0.05 정규성 위배

# mann-whitney u 검정
mannwhitneyu = mannwhitneyu(r_male,r_female)
print(f'mannwhitneyu : {mannwhitneyu[1]:.5f}')
# mannwhitneyu : 0.91737 > 0.05 귀무가설 채택

# 등분산성 검정 - levene
print(f'등분산성 : {stats.levene(r_male, r_female).pvalue:.4f}') # 0.05 보다 크면 등분산성 만족한다.
# 등분산성 : 0.7756 > 0.05 이므로 등분산성 만족

print(f'결과 statistic : {stats.ttest_ind(r_male, r_female, equal_var=True)[0]:.4f}\n'
      f'결과 pvalue : {stats.ttest_ind(r_male, r_female, equal_var=True)[1]:.4f}')
# 결과 statistic : -0.2125
# 결과 pvalue : 0.8332 > 0.05 이므로 귀무가설 채택.
# 결론 : wilcoxon p-value 또한 0.89038 > 0.05 이므로 귀무가설을 채택한다. 
# 성별에 따른 혈관 내의 콜레스테롤 양에 차이가 없다.
'''


'''
[two-sample t 검정 : 문제3]
DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하는지 검정하시오.
* 연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다.
'''
# 귀무가설 : 총무부, 영업부 직원의 연봉의 평균에 차이가 없다.
# 대립가설 : 총무부, 영업부 직원의 연봉의 평균에 차이가 있다.

'''
import sys
import pickle
import MySQLdb

try:
    with open('python_analysis/mymariadb.dat', mode='rb') as obj:
        config = pickle.load(obj) # dbconfig.py의 config를 가져온다
except Exception as e:
    print('읽기 오류 :', e)
    sys.exit() # 오류나면 프로그램 강제 종료

try:
    with MySQLdb.connect(**config) as conn:
        cursor = conn.cursor()

        sql = """
            select busernum, jikwonpay
            from jikwon
        """
        cursor.execute(sql)
        
        df = pd.DataFrame(cursor.fetchall(), columns=['부서번호','연봉'])
        # print(df)
        # df.info() # 30 rows, 

        # 10 - 총무부, 20 - 영업부, 30 - 전산부, 40 - 관리부
        
        chongmu = df[df['부서번호']==10].reset_index()
        youngup = df[df['부서번호']==20].reset_index()
        # print(chongmu)
        # print(youngup)
        # print(f'총무부 평균 연봉 : {chongmu.loc[:,'연봉'].mean():.2f}\n')
        # print(f'영업부 평균 연봉 : {youngup.loc[:,'연봉'].mean():.2f}\n')
        # 총무부 평균 연봉 : 5414.29
        # 영업부 평균 연봉 : 4908.33      

        chongmu_pay = chongmu['연봉']
        youngup_pay = youngup['연봉']

        # 정규성 검정
        # print(len(chongmu), ' ', len(youngup))        
        print(f'총무부 p-value : {stats.shapiro(chongmu_pay).pvalue:.4f}\n') # 0.05 보다 크면 정규성 만족한다.
        # 총무부 p-value : 0.0260 < 0.05 정규성 위배
        print(f'영업부 p-value : {stats.shapiro(youngup_pay).pvalue:.4f}\n')
        # 영업부 p-value : 0.0256 < 0.05 정규성 위배

        # 등분산성 검정 - levene
        print(f'등분산성 : {stats.levene(chongmu_pay, youngup_pay).pvalue:.4f}')
        # 등분산성 : 0.9150 > 0.05 이므로 등분산성 만족

        print(f'결과 statistic : {stats.ttest_ind(chongmu_pay, youngup_pay, equal_var=True).statistic:.4f}'
        f'결과 pvalue : {stats.ttest_ind(chongmu_pay, youngup_pay, equal_var=True).pvalue:.4f}\n')
        # 결과 statistic : 0.4585
        # 결과 pvalue : 0.6524 > 0.05 이므로 귀무가설 채택
        # 총무부, 영업부 직원의 연봉 평균에 차이가 없다.

except Exception as e:
    print(f'오류 발생!! {e}')
'''