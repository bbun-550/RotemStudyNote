'''
## 이원분산 분석
- 두 개의 요인에 대한 집단(독립변수) 각각이 종속변수의 평균에 영향을 주는지 검정한다.
- 가설이 주효과 2개, 교호작용 1개가 나온다.
    - 교호작용이란? (Interaction term) 한 쪽 요인이 취하는 수준에 따라 다른 쪽 요인이 영향을 받는 요인의 조합효과를 말하는 것으로 상승과 상쇄가 있다.
    - eg. 초밥과 간장(상승효과), 감자튀김과 간장, 초밥과 케챱(상쇄효과) ...  두 개의 요인이 영향을 주는 것을 교호작용이라고 한다.

'''
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.rc('font', family='applegothic')
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

'''
#### 실습1
- 태아 수와 관측자 수가 태아의 머리둘레 평균에 영향을 주는가?
- 태아수, 관측자수 - 요인 2가지

- 주효과 가설
- 귀무가설 : 태아 수와 태아의 머리둘레 평균은 차이가 없다.
- 대립가설 : 태아 수와 태아의 머리둘레 평균은 차이가 있다.
- 귀무가설 : 태아 수와 관측자 수의 머리둘레 평균은 차이가 없다.
- 대립가설 : 태아 수와 관측자 수의 머리둘레 평균은 차이가 있다.

- 교호작용 가설
- 귀무가설 : 교호작용이 없다. (태아수와 관측자 수는 관련이 없다)
- 대립가설 : 교호작용이 있다. (태아수와 관측자 수는 관련이 있다)
'''

url = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3_2.txt"
data = pd.read_csv(url)
# print(data.head(3), data.shape) # (36, 3)
#    머리둘레  태아수  관측자수
# 0  14.3    1     1
# 1  14.0    1     1
# 2  14.8    1     1

# print(data['태아수'].unique()) # [1 2 3]
# print(data['관측자수'].unique()) # [1 2 3 4]

# 데이터들이 어떻게 흩어져 있는지 확인하자
# reg = ols("머리둘레 ~ C(태아수) + C(관측자수)", data=data).fit() # 이 방법으로는 교호작용을 확인할 수 없다

# 교호작용이 확인 되는 방법이다.
# reg = ols("머리둘레 ~ C(태아수) + C(관측자수) + C(태아수) :C(관측자수)", data=data).fit()
## 더 짧게 쓰는 법
reg = ols("머리둘레 ~ C(태아수) * C(관측자수)", data=data).fit()

result = anova_lm(reg, typ=2)
# print(result)
#                     sum_sq    df            F        PR(>F)
# C(태아수)          324.008889   2.0  2113.101449  1.051039e-27 < 0.05 귀무기각
# C(관측자수)           1.198611   3.0     5.211353  6.497055e-03 > 0.05 귀무채택
# C(태아수):C(관측자수)    0.562222   6.0     1.222222  3.295509e-01 > 0.05 귀무기각

# 결론 : 태아 수는 머리둘레에 강력한 영향을 미친다. 관측자 수는 유의한 영향을 미친다.
# 하지만 태아수와 관측자 수의 상호작용은 유의하지 않다.

# 주로 one way ANOVA를 쓴다. 

'''
#### 실습2
- poison의 종류와 treat(응급처치)가 독퍼짐 시간의 평균에 영향을 주는가?

- 주효과 가설
- 귀무가설 : poison 종류와 독퍼짐 시간의 평균에 차이가 없다.
- 대립가설 : poison 종류와 독퍼짐 시간의 평균에 차이가 있다.
- 귀무가설 : treat(응급처치) 방법과 독퍼짐 시간의 평균에 차이가 없다.
- 대립가설 : treat(응급처치) 방법과 독퍼짐 시간의 평균에 차이가 있다.

- 교호작용 가설
- 귀무가설 : 교호작용이 없다. (posion 종류와 treat(응급처치) 방법은 관련이 없다.)
- 대립가설 : 교호작용이 있다. (posion 종류와 treat(응급처치) 방법은 관련이 있다.)
'''

data2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/poison_treat.csv', 
                    index_col=0) # 0번째 칼럼을 인덱스로 사용한다.
# print(data2.columns) # ['Unnamed: 0', 'time', 'poison', 'treat']
# print(data2.head(2), data2.shape)
#    time  poison treat
# 1  0.31       1     A
# 2  0.45       1     A (48, 3)

# 데이터 균형 확인
# print(data2.groupby('poison').agg(len)) # 집단 표본 수 확인
# print(data2.groupby('treat').agg(len))
# print(data2.groupby(['poison','treat']).agg(len))
# 모든 집단별 표본 수가 동일하므로 균형설계가 잘 되어있다고 할 수 있다.

reg2 = ols('time ~ C(poison) * C(treat)', data=data2).fit()
result2 = anova_lm(reg2, typ=2)
# print(result2)
#                       sum_sq    df          F        PR(>F)
# C(poison)           1.033012   2.0  23.221737  3.331440e-07 < 0.05 귀무기각
# C(treat)            0.921206   3.0  13.805582  3.777331e-06 < 0.05 귀무기각
# C(poison):C(treat)  0.250137   6.0   1.874333  1.122506e-01 > 0.05 이므로 상호작용 효과는 없다.

# 사후검정 post hoc
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkyResult1 = pairwise_tukeyhsd(endog=data2.time, groups=data2.poison)
# print(turkyResult1)
# Multiple Comparison of Means - Tukey HSD, FWER=0.05 
# ====================================================
# group1 group2 meandiff p-adj   lower   upper  reject
# ----------------------------------------------------
#      1      2  -0.0731 0.5882 -0.2525  0.1063  False
#      1      3  -0.3412 0.0001 -0.5206 -0.1619   True
#      2      3  -0.2681 0.0021 -0.4475 -0.0887   True
# ----------------------------------------------------
# (1, 3),(2, 3) 간에 독 퍼짐 시간 평균의 차이가 있다.

turkyResult2 = pairwise_tukeyhsd(endog=data2.time, groups=data2.treat)
# print(turkyResult2)
# Multiple Comparison of Means - Tukey HSD, FWER=0.05 
# ====================================================
# group1 group2 meandiff p-adj   lower   upper  reject
# ----------------------------------------------------
#      A      B   0.3625  0.001  0.1253  0.5997   True
#      A      C   0.0783 0.8143 -0.1589  0.3156  False
#      A      D     0.22 0.0778 -0.0172  0.4572  False
#      B      C  -0.2842 0.0132 -0.5214 -0.0469   True
#      B      D  -0.1425  0.387 -0.3797  0.0947  False
#      C      D   0.1417 0.3922 -0.0956  0.3789  False
# ----------------------------------------------------
# (A, B), (B, C) treat(응급처치) 방법과 독퍼짐 시간의 평균에 차이가 있다

# 시각화
turkyResult1.plot_simultaneous(xlabel='mean',ylabel='poison')
turkyResult2.plot_simultaneous(xlabel='mean',ylabel='treat')
plt.show()
plt.close()
