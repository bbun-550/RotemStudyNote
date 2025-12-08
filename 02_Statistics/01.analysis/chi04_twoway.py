'''
이원카이제곱 검정
- 동질성 :
    두 집단의 분포가 동일한가? 다른 분포인가? 를 검증하는 방법이다. 두 집단 이상에서 각 범주(집단) 간의 비율이 서로
동일한가를 검정하게 된다. 두 개 이상의 범주형 자료가 동일한 분포를 갖는 모집단에서 추출된 것인지 검정하는 방법이다.

<동질성검정실습1>
교육방법에 따른 교육생들의 만족도 분석 - 동질성검정 survey_method.csv
'''
import pandas as pd
import scipy.stats as stats

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/survey_method.csv')
# print(data.head(2))
#    no  method  survey
# 0   1       1       1
# 1   2       2       2

# 요소 중복성 검사
# print(data['method'].unique()) # [1 2 3]
# print(set(data['survey'])) # {1, 2, 3, 4, 5}

# crosstable 만들고, 칼럼명 인덱스명 바꾸기
ctab = pd.crosstab(index=data['method'], columns=data['survey'])
ctab.columns = ['매우만족','만족','보통','불만족','매우불만족']
ctab.index = ['방법1','방법2','방법3']
# print(ctab)
# survey  1   2   3   4  5
# method
# 1       5   8  15  16  6
# 2       8  14  11  11  6
# 3       8   7  11  15  9

# 방법 별로 contingency 줄 수 있다
# 통으로 contingency 확인
chi2, pvalue, dof, _ = stats.chi2_contingency(ctab)
msg = "test statics : {:.10f}, pvalue : {:.6f}, df : {}"
# print(msg.format(chi2, pvalue, dof))
# test statics : 6.5446678205, pvalue : 0.586457, df : 8

# 결론(해석/평가) : 유의수준 0.05 < 유의확률 0.586457 이므로 귀무가설을 채택한다.
# 검정한 데이터는 우연히 발생된 데이터이다.

'''
동질성

<동질성 검정실습2>
연령대별 sns 이용률의 동질성 검정
20대에서 40대까지 연령대별로 서로 조금씩 그 특성이 다른 SNS 서비스들에 대해 이용현황을 조사한 자료를 바탕으로 연령대별로 홍보
전략을 세우고자 한다.
연령대별로 이용현황이 서로 동일한지 검정해 보도록 하자.
'''

# 가설
# 귀무가설 : 연령대별로 sns 서비스별 이용 현황은 동일하다.
# 대립가설 : 연령대별로 sns 서비스별 이용 현황은 동일하지 않다. 

data2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/snsbyage.csv')
# print(data2.head(2))
#    age service
# 0    1       F
# 1    1       F

# print(data2['age'].unique()) # [1 2 3]
# print(data2['service'].unique()) # ['F' 'T' 'K' 'C' 'E']

ctab2 = pd.crosstab(index=data2['age'],columns=data2['service'])
# print(ctab2)
# service    C   E    F    K    T
# age
# 1         81  16  207  111  117
# 2        109  15  107  236  104
# 3         32  17   78  133   76

chi2, pvalue, dof, _ = stats.chi2_contingency(ctab2)
msg = "test statics : {:.10f}, pvalue : {:.6f}, df : {}"
# print(msg.format(chi2, pvalue, dof))
# test statics : 102.7520249448, pvalue : 0.000000, df : 8

# 해석 : 유의수준 0.05 > 유의확률 이므로 귀무가설을 기각한다.

'''
사실 위 데이터는 샘플 데이터이다. 
그런데 샘플링 연습을 위해 위 데이터를 모집이라고 가정하자.
표본을 추출해서 처리해보자.
'''
sample_data = data2.sample(n=50, replace=True, random_state=1) # random_state : seed처럼 random을 고정하는 기능
print(sample_data.head(2))

ctab3 = pd.crosstab(index=sample_data['age'],columns=sample_data['service'])
print(ctab3)
# service  C  E  F  K  T
# age
# 1        5  0  3  3  5
# 2        3  2  5  9  3
# 3        3  0  1  6  2

chi2, pvalue, dof, _ = stats.chi2_contingency(ctab3)
msg = "test statics : {:.10f}, pvalue : {:.6f}, df : {}"
print(msg.format(chi2, pvalue, dof))
# test statics : 8.4651438629, pvalue : 0.389400, df : 8

