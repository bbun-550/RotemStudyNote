#가구 유아 수에 따른 품목별 소비액 치이가 있다/없다(ANOVA)
    # - h0 : 가구별 유아 수에 따른 품목 소비량 차이가 없다.
    # - h1 : 가구별 유아 수에 따른 품목 소비량 차이가 있다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font' , family='applegothic')
import seaborn
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

# 사후검정 필요

import seaborn as sns
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
cols = ["MntWines","MntFruits","MntMeatProducts",
        "MntFishProducts","MntSweetProducts","MntGoldProds"]

for i, col in enumerate(cols):
    ax = axes[i//3, i%3]
    sns.boxplot(x='Kidhome', y=col, data=data, ax=ax, palette='Paired')
    ax.set_title(f"Kidhome별 {col} 소비 분포")
    ax.set_xlabel('가구내 유아 수')
    ax.set_ylabel('소비량')

plt.tight_layout()
plt.show()


cols = ["MntWines","MntFruits","MntMeatProducts",
        "MntFishProducts","MntSweetProducts","MntGoldProds"]

# 가구당 유아 수 품목별 평균 
pivot_mean = data.groupby('Kidhome')[cols].mean().round(1)

plt.figure(figsize=(9, 4.8))
sns.heatmap(pivot_mean, annot=True, fmt=".1f", cmap="Blues",
            cbar_kws={"label": "평균 소비량"}, linewidths=.5, linecolor="white")
plt.title("가구당 유아 수 × 품목별 평균 소비량")
plt.xlabel("품목")
plt.ylabel("가구당 유아 수")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import scikit_posthocs as sp
# Dunn’s test 실행
dunn_result = sp.posthoc_dunn(data, val_col='MntWines', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)
#                0              1             2
# 0   1.000000e+00  1.055980e-159  1.528503e-16   0과 비교한 1, 2는 p값이 0에 근사함
# 1  1.055980e-159   1.000000e+00  1.000000e+00   1과2 비교했을때 p값이 1나왔다
# 2   1.528503e-16   1.000000e+00  1.000000e+00
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="Blues", linewidths=0.5)
plt.title("Dunn's test 사후검정(와인)")
plt.xlabel("가구 당 유아 수")
plt.ylabel("와인 구매액")
plt.show()

dunn_result = sp.posthoc_dunn(data, val_col='MntFruits', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)
#               0             1             2
# 0  1.000000e+00  1.002681e-92  2.820945e-16   0과 비교한 1, 2는 p값이 0에 근사함
# 1  1.002681e-92  1.000000e+00  5.863554e-02   1과2 비교했을 때 p값이 0.058
# 2  2.820945e-16  5.863554e-02  1.000000e+00
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="Blues", linewidths=0.5)
plt.title("Dunn's test 사후검정(과일)")
plt.xlabel("가구 당 유아 수")
plt.ylabel("과일 구매액")
plt.show()

dunn_result = sp.posthoc_dunn(data, val_col='MntMeatProducts', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)
#                0              1             2
# 0   1.000000e+00  6.898412e-141  2.661294e-18   0과 비교한 1, 2는 p값이 0에 근사함
# 1  6.898412e-141   1.000000e+00  4.236794e-01   1과2 비교했을 때 p값이 0.42
# 2   2.661294e-18   4.236794e-01  1.000000e+00
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="Blues", linewidths=0.5)
plt.title("Dunn's test 사후검정(육류)")
plt.xlabel("가구 당 유아 수")
plt.ylabel("육류 구매액")
plt.show()

dunn_result = sp.posthoc_dunn(data, val_col='MntFishProducts', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)
#               0             1             2
# 0  1.000000e+00  7.096210e-95  4.348503e-14   0과 비교한 1, 2는 p값이 0에 근사함
# 1  7.096210e-95  1.000000e+00  2.966549e-01   1과2 비교했을 때 p값이 0.3
# 2  4.348503e-14  2.966549e-01  1.000000e+00
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="Blues", linewidths=0.5)
plt.title("Dunn's test 사후검정(어류)")
plt.xlabel("가구 당 유아 수")
plt.ylabel("어류 구매액")
plt.show()

dunn_result = sp.posthoc_dunn(data, val_col='MntSweetProducts', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)
#               0             1             2
# 0  1.000000e+00  2.529004e-88  4.457389e-16   0과 비교한 1, 2는 p값이 0에 근사함
# 1  2.529004e-88  1.000000e+00  4.589095e-02   1과2 비교했을 때 p값이 0.45
# 2  4.457389e-16  4.589095e-02  1.000000e+00
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="Blues", linewidths=0.5)
plt.title("Dunn's test 사후검정(디저트)")
plt.xlabel("가구 당 유아 수")
plt.ylabel("디저트류 구매액")
plt.show()

dunn_result = sp.posthoc_dunn(data, val_col='MntGoldProds', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)
#               0             1             2
# 0  1.000000e+00  3.665120e-84  3.060763e-12   0과 비교한 1, 2는 p값이 0에 근사함
# 1  3.665120e-84  1.000000e+00  4.463550e-01   1과2 비교했을 때 p값이 0.44
# 2  3.060763e-12  4.463550e-01  1.000000e+00
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="Blues", linewidths=0.5)
plt.title("Dunn's test 사후검정(할인 혜택이 없는 상품)")
plt.xlabel("가구 당 유아 수")
plt.ylabel("할인 혜택이 없는 상품 구매액")
plt.show()