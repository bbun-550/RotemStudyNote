#가구 유아 수에 따른 품목별 소비액 치이가 있다/없다(ANOVA)
    # - h0 : 가구별 유아 수에 따른 품목 소비량 차이가 없다.
    # - h1 : 가구별 유아 수에 따른 품목 소비량 차이가 있다.
#-------------> 정규성 불만족으로인한 아노바 검정 불가



# Kruskal-Wallis 으로 검정시 귀무, 대립 가설 수정필요
#     # - h0 : 가구별 유아 수에 따른 품목 소비량 분포에 차이가 없다.
#     # - h1 : 가구별 유아 수에 따른 품목 소비량 분포에 차이가 있다.


# Kruskal-Wallis 검정을 사용할 때 가설을 바꿔야 하는 이유는 이 검정의 통계적 특성과 귀무가설의 정의가 다르기 때문입니다.
# 같은 "차이 검정"이지만, 어떤 값의 차이를 보느냐가 달라요.

# ✅ 정리: 왜 가설을 바꿔야 하나요?
# 검정 방법	귀무가설 (H₀)	대립가설 (H₁)
# ANOVA (F-test)	그룹 간 평균이 모두 같다	그룹 간 평균에 차이가 있다
# Kruskal-Wallis	그룹 간 분포의 위치(중앙값) 가 같다	하나 이상의 그룹이 다른 분포를 가진다

# → 즉, Kruskal-Wallis는 평균이 아닌 "중앙값이나 순위" 차이를 검정합니다.

# 🔍 예를 들어

# ANOVA는 데이터를 정규분포라고 가정하고, 평균 간 차이를 봅니다.

# Kruskal-Wallis는 정규성 가정을 하지 않고, 데이터 순위 기반으로 그룹 간 차이가 있는지를 봅니다.

# 그래서 ANOVA에서 쓰던 가설 문장을 그대로 쓰면 통계적으로 틀린 해석이 될 수 있어요.

# 📌 가설 문장 차이 예시
# ❌ ANOVA에서 쓰던 가설 (Kruskal-Wallis에서는 부적절)

# H₀: 유아 수에 따른 평균 소비량에 차이가 없다.
# H₁: 유아 수에 따라 평균 소비량에 차이가 있다.

# ✅ Kruskal-Wallis에 맞는 가설

# H₀: 유아 수에 따라 소비량 분포(또는 중앙값)에 차이가 없다.
# H₁: 유아 수에 따라 소비량 분포(또는 중앙값)에 차이가 있다.

# ✅ 결론 요약
# 항목	설명
# Kruskal-Wallis는	비모수 검정 → 평균이 아니라 중앙값/순위 비교
# 가설을 바꿔야 하는 이유	비교 대상이 평균이 아니라 분포(또는 중앙값) 이기 때문
# 언제 쓰나요?	정규성 또는 등분산성이 없는 경우 (비정규, 이상치 많은 데이터)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font' , family='malgun gothic')
import seaborn as sns
import statsmodels.formula.api as smf


data = pd.read_csv('filtered_MarketingData.csv')
# Kidhome
# cols = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]


# 관계검정과 차이검정 어떤것을 적용할것이냐
# 가구 유아수 0, 1 ,2 3개 니까 아노바로 한다 - 범주형 독립변수
# 독립변수를 유아수  종속변수로 품목 소비량 

# "유아 수별 소비 평균이 다르냐?"
# ✅ ANOVA

# 있다면
# "유아 수가 늘어나면 소비가 얼마나 증가하냐?"
# ✅ 선형회귀

from scipy.stats import f_oneway

group0 = data[data['Kidhome'] == 0]['MntWines']
group1 = data[data['Kidhome'] == 1]['MntWines']
group2 = data[data['Kidhome'] == 2]['MntWines']

f_static , p_value = f_oneway(group0, group1, group2)
print(f' MntWines   f-static : {f_static}, p-value : {p_value}')


group0 = data[data['Kidhome'] == 0]['MntFruits']
group1 = data[data['Kidhome'] == 1]['MntFruits']
group2 = data[data['Kidhome'] == 2]['MntFruits']
f_static , p_value = f_oneway(group0, group1, group2)
print(f' MntFruits   f-static : {f_static}, p-value : {p_value}')

group0 = data[data['Kidhome'] == 0]['MntMeatProducts']
group1 = data[data['Kidhome'] == 1]['MntMeatProducts']
group2 = data[data['Kidhome'] == 2]['MntMeatProducts']
f_static , p_value = f_oneway(group0, group1, group2)
print(f' MntMeatProducts   f-static : {f_static}, p-value : {p_value}')

group0 = data[data['Kidhome'] == 0]['MntFishProducts']
group1 = data[data['Kidhome'] == 1]['MntFishProducts']
group2 = data[data['Kidhome'] == 2]['MntFishProducts']
f_static , p_value = f_oneway(group0, group1, group2)
print(f' MntFishProducts   f-static : {f_static}, p-value : {p_value}')


group0 = data[data['Kidhome'] == 0]['MntSweetProducts']
group1 = data[data['Kidhome'] == 1]['MntSweetProducts']
group2 = data[data['Kidhome'] == 2]['MntSweetProducts']

f_static , p_value = f_oneway(group0, group1, group2)
print(f' MntSweetProducts   f-static : {f_static}, p-value : {p_value}')

group0 = data[data['Kidhome'] == 0]['MntGoldProds']
group1 = data[data['Kidhome'] == 1]['MntGoldProds']
group2 = data[data['Kidhome'] == 2]['MntGoldProds']

f_static , p_value = f_oneway(group0, group1, group2)
print(f' MntGoldProds   f-static : {f_static}, p-value : {p_value}')


#  MntWines   f-static : 392.7413052360142, p-value : 1.072473498978659e-146
#  MntFruits   f-static : 192.07099135769002, p-value : 1.2090676283177211e-77
#  MntMeatProducts   f-static : 284.183704088864, p-value : 1.4127252512438615e-110
#  MntFishProducts   f-static : 210.0139548010787, p-value : 3.0758501685519866e-84
#  MntSweetProducts   f-static : 195.66408877184134, p-value : 5.683705894827886e-79
#  MntGoldProds   f-static : 169.29109565547785, p-value : 3.865038076879511e-69

# 모두 pvalue <0.05 유의미한 차이가 있다

# 여기까지만하면 차이 있는지만 알고 사후검정이 필요하다

# 3개 이상 집단 평균 비교시 절차
# 정규성 + 등분산성 가정 충족여부
# 모두 만족 시 일원 분산 분석 
# 등분산성 불만족 웰치스 아노바
# 정규성 불만족 비모수 검정

# 사후검정 Post-hoc test

from scipy.stats import shapiro

stat, p = shapiro(data.MntWines)
print("p-value:", p)

if p > 0.05:
    print("정규성을 만족함")
else:
    print("정규성을 만족하지 않음")




stat, p = shapiro(data.MntFruits)
print("p-value:", p)

if p > 0.05:
    print("정규성을 만족함")
else:
    print("정규성을 만족하지 않음")



stat, p = shapiro(data.MntMeatProducts)
print("p-value:", p)

if p > 0.05:
    print("정규성을 만족함")
else:
    print("정규성을 만족하지 않음")




stat, p = shapiro(data.MntFishProducts)
print("p-value:", p)

if p > 0.05:
    print("정규성을 만족함")
else:
    print("정규성을 만족하지 않음")




stat, p = shapiro(data.MntSweetProducts)
print("p-value:", p)

if p > 0.05:
    print("정규성을 만족함")
else:
    print("정규성을 만족하지 않음")




stat, p = shapiro(data.MntGoldProds)
print("p-value:", p)

if p > 0.05:
    print("정규성을 만족함")
else:
    print("정규성을 만족하지 않음")


# p-value: 1.0004362799631704e-42
# 정규성을 만족하지 않음
# p-value: 2.668172028444469e-53
# 정규성을 만족하지 않음
# p-value: 1.0186513551965282e-50
# 정규성을 만족하지 않음
# p-value: 3.2352531611067182e-52
# 정규성을 만족하지 않음
# p-value: 2.253404708479899e-53
# 정규성을 만족하지 않음
# p-value: 2.415728553234845e-48
# 정규성을 만족하지 않음


print(data.head(1))
#    MntWines  MntFruits  MntMeatProducts  MntFishProducts  MntSweetProducts  MntGoldProds   Income  Kidhome   
# 0       635         88              546              172                88            88  58138.0        0 



from scipy.stats import kruskal
group0 = data[data['Kidhome'] == 0]['MntWines']
group1 = data[data['Kidhome'] == 1]['MntWines']
group2 = data[data['Kidhome'] == 2]['MntWines']
stat, p = kruskal(group0, group1, group2)

print("Kruskal-Wallis H 통계량:", stat)
print("p-value:", p)

if p < 0.05:
    print("→ 유의미한 차이 있음 (귀무가설 기각)")
else:
    print("→ 유의미한 차이 없음 (귀무가설 채택)")



group0 = data[data['Kidhome'] == 0]['MntFruits']
group1 = data[data['Kidhome'] == 1]['MntFruits']
group2 = data[data['Kidhome'] == 2]['MntFruits']
stat, p = kruskal(group0, group1, group2)

print("Kruskal-Wallis H 통계량:", stat)
print("p-value:", p)

if p < 0.05:
    print("→ 유의미한 차이 있음 (귀무가설 기각)")
else:
    print("→ 유의미한 차이 없음 (귀무가설 채택)")

group0 = data[data['Kidhome'] == 0]['MntMeatProducts']
group1 = data[data['Kidhome'] == 1]['MntMeatProducts']
group2 = data[data['Kidhome'] == 2]['MntMeatProducts']
stat, p = kruskal(group0, group1, group2)

print("Kruskal-Wallis H 통계량:", stat)
print("p-value:", p)

if p < 0.05:
    print("→ 유의미한 차이 있음 (귀무가설 기각)")
else:
    print("→ 유의미한 차이 없음 (귀무가설 채택)")

group0 = data[data['Kidhome'] == 0]['MntFishProducts']
group1 = data[data['Kidhome'] == 1]['MntFishProducts']
group2 = data[data['Kidhome'] == 2]['MntFishProducts']
stat, p = kruskal(group0, group1, group2)

print("Kruskal-Wallis H 통계량:", stat)
print("p-value:", p)

if p < 0.05:
    print("→ 유의미한 차이 있음 (귀무가설 기각)")
else:
    print("→ 유의미한 차이 없음 (귀무가설 채택)")


group0 = data[data['Kidhome'] == 0]['MntSweetProducts']
group1 = data[data['Kidhome'] == 1]['MntSweetProducts']
group2 = data[data['Kidhome'] == 2]['MntSweetProducts']
stat, p = kruskal(group0, group1, group2)

print("Kruskal-Wallis H 통계량:", stat)
print("p-value:", p)

if p < 0.05:
    print("→ 유의미한 차이 있음 (귀무가설 기각)")
else:
    print("→ 유의미한 차이 없음 (귀무가설 채택)")


group0 = data[data['Kidhome'] == 0]['MntGoldProds']
group1 = data[data['Kidhome'] == 1]['MntGoldProds']
group2 = data[data['Kidhome'] == 2]['MntGoldProds']
stat, p = kruskal(group0, group1, group2)

print("Kruskal-Wallis H 통계량:", stat)
print("p-value:", p)

if p < 0.05:
    print("→ 유의미한 차이 있음 (귀무가설 기각)")
else:
    print("→ 유의미한 차이 없음 (귀무가설 채택)")

# Kruskal-Wallis H 통계량: 754.4502697944214
# p-value: 1.4900659099023457e-164
# → 유의미한 차이 있음 (귀무가설 기각)
# Kruskal-Wallis H 통계량: 454.39832898203537
# p-value: 2.1313588102151002e-99
# → 유의미한 차이 있음 (귀무가설 기각)
# Kruskal-Wallis H 통계량: 675.3132451028933
# p-value: 2.2782010506932533e-147
# → 유의미한 차이 있음 (귀무가설 기각)
# Kruskal-Wallis H 통계량: 456.9705179992229
# p-value: 5.889966439639059e-100
# → 유의미한 차이 있음 (귀무가설 기각)
# Kruskal-Wallis H 통계량: 434.23621466532904
# p-value: 5.091011336904064e-95
# → 유의미한 차이 있음 (귀무가설 기각)
# Kruskal-Wallis H 통계량: 403.4946802934537
# p-value: 2.411256634590874e-88
# → 유의미한 차이 있음 (귀무가설 기각)


import scikit_posthocs as sp

# MntWines  MntFruits  MntMeatProducts  
#MntFishProducts  MntSweetProducts  MntGoldProds   Income  Kidhome 

# Dunn’s test 실행
dunn_result = sp.posthoc_dunn(data, val_col='MntWines', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)

#                0              1             2
# 0   1.000000e+00  1.055980e-159  1.528503e-16
# 1  1.055980e-159   1.000000e+00  1.000000e+00
# 2   1.528503e-16   1.000000e+00  1.000000e+00
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="coolwarm", linewidths=0.5)
plt.title("Dunn's Test Pairwise p-values (Bonferroni corrected)")
plt.xlabel("Group")
plt.ylabel("Group")
plt.show()


# - `group_col='Kidhome'` 이므로, 집단은 **유아 수(Kidhome)** 에 따라 나뉩니다.
# - Kidhome 값이 0, 1, 2인 가구로 나뉘는 거죠.

# ---

# ## 🔍 결과 해석

# 결과 표:

# |     | 0 (유아 0명) | 1 (유아 1명) | 2 (유아 2명) |
# |-----|---------------|---------------|---------------|
# | 0   | 1.000         | 1.056e-159     | 1.529e-16     |
# | 1   | 1.056e-159     | 1.000         | 1.000         |
# | 2   | 1.529e-16     | 1.000         | 1.000         |

# - 이 표는 **각 집단 간 pairwise p-value**를 보여줍니다.
# - 예를 들어:
#   - `0 vs 1`: p = 1.056e-159 → **유의미한 차이 있음** (`p < 0.05`)
#   - `0 vs 2`: p = 1.529e-16 → **유의미한 차이 있음**
#   - `1 vs 2`: p = 1.000 → **차이 없음**

# ---

# ## 📌 요약

# | 숫자 | 의미 |
# |------|------|
# | `0` | `Kidhome == 0`인 그룹 (유아 없는 가구) |
# | `1` | `Kidhome == 1`인 그룹 (유아 1명 가구) |
# | `2` | `Kidhome == 2`인 그룹 (유아 2명 가구) |

# ---

# ## 🔎 해석 예

# > "유아가 없는 가구(0명)와 유아가 1명 또는 2명 있는 가구 간에는 **소비량(MntWines)에 유의한 차이**가 있지만,  
# > 유아 1명과 유아 2명 있는 가구 사이에는 **차이가 없다**"라고 해석할 수 있습니다.

# ---


dunn_result = sp.posthoc_dunn(data, val_col='MntFruits', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)

#               0             1             2
# 0  1.000000e+00  1.002681e-92  2.820945e-16
# 1  1.002681e-92  1.000000e+00  5.863554e-02
# 2  2.820945e-16  5.863554e-02  1.000000e+00
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="coolwarm", linewidths=0.5)
plt.title("Dunn's Test Pairwise p-values (Bonferroni corrected)")
plt.xlabel("Group")
plt.ylabel("Group")
plt.show()



dunn_result = sp.posthoc_dunn(data, val_col='MntMeatProducts', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)


#                0              1             2
# 0   1.000000e+00  6.898412e-141  2.661294e-18
# 1  6.898412e-141   1.000000e+00  4.236794e-01
# 2   2.661294e-18   4.236794e-01  1.000000e+00
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="coolwarm", linewidths=0.5)
plt.title("Dunn's Test Pairwise p-values (Bonferroni corrected)")
plt.xlabel("Group")
plt.ylabel("Group")
plt.show()




dunn_result = sp.posthoc_dunn(data, val_col='MntFishProducts', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)



#               0             1             2
# 0  1.000000e+00  7.096210e-95  4.348503e-14
# 1  7.096210e-95  1.000000e+00  2.966549e-01
# 2  4.348503e-14  2.966549e-01  1.000000e+00
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="coolwarm", linewidths=0.5)
plt.title("Dunn's Test Pairwise p-values (Bonferroni corrected)")
plt.xlabel("Group")
plt.ylabel("Group")
plt.show()



dunn_result = sp.posthoc_dunn(data, val_col='MntSweetProducts', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)



#               0             1             2
# 0  1.000000e+00  2.529004e-88  4.457389e-16
# 1  2.529004e-88  1.000000e+00  4.589095e-02
# 2  4.457389e-16  4.589095e-02  1.000000e+00
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="coolwarm", linewidths=0.5)
plt.title("Dunn's Test Pairwise p-values (Bonferroni corrected)")
plt.xlabel("Group")
plt.ylabel("Group")
plt.show()




dunn_result = sp.posthoc_dunn(data, val_col='MntGoldProds', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)

#               0             1             2
# 0  1.000000e+00  3.665120e-84  3.060763e-12
# 1  3.665120e-84  1.000000e+00  4.463550e-01
# 2  3.060763e-12  4.463550e-01  1.000000e+00




plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="coolwarm", linewidths=0.5)
plt.title("Dunn's Test Pairwise p-values (Bonferroni corrected)")
plt.xlabel("Group")
plt.ylabel("Group")
plt.show()


# ✅ 해석 기준  정리

# p < 0.05 → 그룹 간 유의미한 차이 있음 (귀무가설 기각)

# p ≥ 0.05 → 그룹 간 차이 없음 (귀무가설 채택)

# 여기서 각 표의 행/열 번호 0, 1, 2는 Kidhome 값 (유아 수) 의미

# ✅ 6개 결과 요약
# #	비교 그룹	유아 수 (Kidhome)	유의미한 차이 있음?	비고
# 1	0 vs 1	없음 vs 1명	✅ 있음 (p ≈ 1e-159)	매우 유의미함
# 	0 vs 2	없음 vs 2명	✅ 있음 (p ≈ 1e-16)	유의미함
# 	1 vs 2	1명 vs 2명	❌ 없음 (p = 1.0)	—
# ---	------------	-------------------	----------------------	------
# 2	0 vs 1	없음 vs 1명	✅ 있음 (p ≈ 1e-92)	매우 유의미
# 	0 vs 2	없음 vs 2명	✅ 있음 (p ≈ 1e-16)	유의미
# 	1 vs 2	1명 vs 2명	❌ 없음 (p = 0.0586)	거의 경계선
# ---	------------	-------------------	----------------------	------
# 3	0 vs 1	없음 vs 1명	✅ 있음 (p ≈ 1e-141)	매우 유의미
# 	0 vs 2	없음 vs 2명	✅ 있음 (p ≈ 1e-18)	유의미
# 	1 vs 2	1명 vs 2명	❌ 없음 (p = 0.423)	—
# ---	------------	-------------------	----------------------	------
# 4	0 vs 1	없음 vs 1명	✅ 있음 (p ≈ 1e-95)	매우 유의미
# 	0 vs 2	없음 vs 2명	✅ 있음 (p ≈ 1e-14)	유의미
# 	1 vs 2	1명 vs 2명	❌ 없음 (p = 0.297)	—
# ---	------------	-------------------	----------------------	------
# 5	0 vs 1	없음 vs 1명	✅ 있음 (p ≈ 1e-88)	매우 유의미
# 	0 vs 2	없음 vs 2명	✅ 있음 (p ≈ 1e-16)	유의미
# 	1 vs 2	1명 vs 2명	✅ 있음 (p = 0.0459)	유일하게 유의미!
# ---	------------	-------------------	----------------------	------
# 6	0 vs 1	없음 vs 1명	✅ 있음 (p ≈ 1e-84)	매우 유의미
# 	0 vs 2	없음 vs 2명	✅ 있음 (p ≈ 1e-12)	유의미
# 	1 vs 2	1명 vs 2명	❌ 없음 (p = 0.446)	—
# 📌 종합 요약

# ✅ Kidhome=0 (유아 없음) 은 항상 다른 그룹과 유의미한 차이 있음

# ❌ Kidhome=1 vs 2 는 대부분 차이 없음, 단 1건만 유의미 (p = 0.0459)

# 전체적으로 보면, 유아가 있는 집 vs 없는 집 간 소비 차이가 크다는 경향이 보임

# 📈 추가 팁

# 이제 각 분석이 어떤 소비 항목(MntWines, MntFruits 등)에 대한 것인지 구체적으로 정리하면:

# 분석 번호	소비 항목	차이 해석
# 1	예: MntWines	없음 vs 있음 간 큰 차이
# 2	예: MntFruits	1 vs 2명 차이 거의 유의미
# 3~6	다른 소비 항목들	대부분 0 vs 나머지 차이 명확

# → 이 정보를 기반으로 보고서, 발표 자료 또는 정책 제안(유아 수와 소비 패턴의 연관성 등)을 만들 수 있습니다.


















