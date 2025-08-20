import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd


'''
## ANOVA 예제1
- 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
- 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.
- 조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.

- 가설
- 귀무가설 : 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 없다.
- 대립가설 : 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 있다.
'''

'''
raw = {'kind':[
    1,
    2,
    3,
    4,
    2,
    1,
    3,
    4,
    2,
    1,
    2,
    3,
    4,
    1,
    2,
    1,
    1,
    3,
    4,
    2
],'quantity':[
    64,
    72,
    68,
    77,
    56,
    'NaN',
    95,
    78,
    55,
    91,
    63,
    49,
    70,
    80,
    90,
    33,
    44,
    55,
    66,
    77
]}
data = pd.DataFrame(raw)
# print(data)
#     kind quantity
# 0      1       64
# 1      2       72

# NaN을 숫자 평균으로 대체 (문자열 아님)
data['quantity'] = pd.to_numeric(data['quantity'], errors='coerce')
# print(data)
data['quantity'] = data['quantity'].fillna(data['quantity'].mean())

oil1 = data.loc[data['kind'] == 1, 'quantity']
oil2 = data.loc[data['kind'] == 2, 'quantity']
oil3 = data.loc[data['kind'] == 3, 'quantity']
oil4 = data.loc[data['kind'] == 4, 'quantity']

print(f'oil1 평균: {np.mean(oil1):.4f}\n' 
      f'oil2 평균: {np.mean(oil2):.4f}\n'
      f'oil3 평균: {np.mean(oil3):.4f}\n'
      f'oil4 평균: {np.mean(oil4):.4f}\n')
# oil1 평균: 63.2500
# oil2 평균: 68.8333
# oil3 평균: 66.7500
# oil4 평균: 72.7500
# 평균들에 차이가 있다고 or 없다고 볼 것인가?

# 정규성 검정
print(f'oil1 정규성 : {stats.shapiro(oil1).pvalue:.6f}')
print(f'oil2 정규성 : {stats.shapiro(oil2).pvalue:.6f}') 
print(f'oil3 정규성 : {stats.shapiro(oil3).pvalue:.6f}') 
print(f'oil4 정규성 : {stats.shapiro(oil4).pvalue:.6f}') 
# oil1 정규성 : 0.868026
# oil2 정규성 : 0.592392
# oil3 정규성 : 0.486011
# oil4 정규성 : 0.416216
# 모두 0.05 보다 크므로 정규성을 만족한다. + 표본 크기가 작으면 정규성 검정의 검정력이 낮기 때문에 Q-Q plot 같은 시각화도 보조로 활용하는 게 좋다

# 등분산성 검정
print(f'등분산성 levene : {stats.levene(oil1, oil2, oil3, oil4).pvalue:.4f}')
# 등분산성 levene : 0.3273 > 0.05 이므로 등분산성 만족한다.
# 정규성, 등분산성 만족하므로 ANOVA 검정을 실시한다.

# ANOVA 검정
print(f'ANOVA p-value : {stats.f_oneway(oil1, oil2, oil3, oil4).pvalue:.6f}')
# ANOVA p-value : 0.848052 > 0.05 이므로 귀무가설 채택한다.
# 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 없다.

# 사후검정
# 진행하는 의미 없다.
turkyResult = pairwise_tukeyhsd(endog=data.quantity, groups=data.kind)
# print(turkyResult)
#  Multiple Comparison of Means - Tukey HSD, FWER=0.05 
# =====================================================
# group1 group2 meandiff p-adj   lower    upper  reject
# -----------------------------------------------------
#      1      2   5.5833 0.9399  -22.489 33.6556  False
#      1      3      3.5 0.9883 -27.8858 34.8858  False
#      1      4      9.5 0.8221 -21.8858 40.8858  False
#      2      3  -2.0833 0.9975 -33.4691 29.3025  False
#      2      4   3.9167 0.9838 -27.4691 35.3025  False
#      3      4      6.0 0.9581 -28.3814 40.3814  False
# -----------------------------------------------------

# 시각화
# turkyResult.plot_simultaneous(xlabel='mean',ylabel='group')
# plt.show()
# plt.close()
'''


'''
## ANOVA 예제2
- DB에 저장된 buser와 jikwon 테이블을 이용하여 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 있는지 검정하시오. 
- 만약에 연봉이 없는 직원이 있다면 작업에서 제외한다.

- 가설
- 귀무가설 : 부서별 직원 연봉의 평균에 차이가 없다.
- 대립가설 : 부서별 직원 연봉의 평균에 차이가 있다.
'''

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
        jeonsan = df[df['부서번호']==30].reset_index()
        gwanli = df[df['부서번호']==40].reset_index()
        # print(chongmu)
        # print(youngup)
        print(f'총무부 평균 연봉 : {chongmu.loc[:,'연봉'].mean():.2f}')
        print(f'영업부 평균 연봉 : {youngup.loc[:,'연봉'].mean():.2f}')
        print(f'전산부 평균 연봉 : {jeonsan.loc[:,'연봉'].mean():.2f}')
        print(f'관리부 평균 연봉 : {gwanli.loc[:,'연봉'].mean():.2f}')
        # 총무부 평균 연봉 : 5414.29
        # 영업부 평균 연봉 : 4908.33
        # 전산부 평균 연봉 : 5328.57
        # 관리부 평균 연봉 : 6262.50
        # 평균의 차이가 유의미한가?

        chongmu_pay = chongmu['연봉']
        youngup_pay = youngup['연봉']
        jeonsan_pay = jeonsan['연봉']
        gwanli_pay = gwanli['연봉']
        
        # 정규성 검정
        # print(len(chongmu), ' ', len(youngup))        
        print(f'총무부 정규성 p-value : {stats.shapiro(chongmu_pay).pvalue:.4f}') # 0.05 보다 크면 정규성 만족한다.        
        print(f'영업부 정규성 p-value : {stats.shapiro(youngup_pay).pvalue:.4f}')
        print(f'전산부 정규성 p-value : {stats.shapiro(jeonsan_pay).pvalue:.4f}')
        print(f'관리부 정규성 p-value : {stats.shapiro(gwanli_pay).pvalue:.4f}')
        # 총무부 정규성 p-value : 0.0260
        # 영업부 정규성 p-value : 0.0256
        # 전산부 정규성 p-value : 0.4194
        # 관리부 정규성 p-value : 0.9078
        # 총무부, 영업부는 정규성을 위배하고 전산부,관리부는 정규성을 만족한다. 정규성 보류

        # 등분산성 검정 - levene
        print(f'levene p-value : {stats.levene(chongmu_pay, youngup_pay, jeonsan_pay, gwanli_pay).pvalue:.4f}')
        # levene p-value : 0.7981 > 0.05 이므로 등분산성 만족

        # 정규성 보류, 등분산성 만족이므로 ANOVA 검정도 실시하고, Kruskal-Wallis 검정도 실시한다

        # ANOVA 검정
        print(f'ANOVA p-value : {stats.f_oneway(chongmu_pay, youngup_pay, jeonsan_pay, gwanli_pay).pvalue:.6f}')
        # ANOVA p-value : 0.745442 > 0.05 이므로 귀무가설 채택한다.

        # Kruskal-Wallis 검정
        print(f'Kruskal-Wallis p-value : {stats.kruskal(chongmu_pay, youngup_pay, jeonsan_pay, gwanli_pay).pvalue:.6f}')
        # Kruskal-Wallis p-value : 0.643344 > 0.05 이므로 귀무가설 채택한다.
        # 정규성을 만족한다고 가정하고 ANOVA 검정했을 때와 정규성을 위배한다고 가정하고 Kruskal 검정을 실시했을 때 pvalue > 0.05 이므로
        # 귀무가설을 채택한다. 즉, 부서별 연봉 평균에 차이는 없다고 볼 수 있다.

except Exception as e:
    print(f'오류 발생!! {e}')

# 사후 검정
# 진행하는 의미 없음
turkyResult = pairwise_tukeyhsd(endog=df['연봉'], groups=df['부서번호'])
# print(turkyResult)
#    Multiple Comparison of Means - Tukey HSD, FWER=0.05    
# ==========================================================
# group1 group2  meandiff p-adj    lower      upper   reject
# ----------------------------------------------------------
#     10     20 -505.9524 0.9588 -3292.2114 2280.3066  False
#     10     30  -85.7143 0.9998  -3217.199 3045.7705  False
#     10     40  848.2143 0.9202 -2823.7771 4520.2056  False
#     20     30  420.2381 0.9756 -2366.0209 3206.4971  False
#     20     40 1354.1667 0.6937 -2028.2234 4736.5568  False
#     30     40  933.9286  0.897 -2738.0628 4605.9199  False
# ----------------------------------------------------------

# 시각화
# turkyResult.plot_simultaneous(xlabel='mean',ylabel='group')
# plt.show()
# plt.close()

# ![[ANOVA예제2_사후검정시각화.png]]
'''

'''
#### 다른 방법 참고
df = pd.read_sql(sql, conn) # v
df.columns = ['부서명', '연봉']
print(df)
buser_group = df.groupby('부서명')['연봉'].mean()
print(buser_group)

gwanli = df[df['부서명'] == '관리부']['연봉']
yeongup = df[df['부서명'] == '영업부']['연봉']
jeonsan = df[df['부서명'] == '전산부']['연봉']
chongmu = df[df['부서명'] == '총무부']['연봉']
'''