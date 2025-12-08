'''
이원카이제곱 검정(교차분할표 사용)
- 변인이 2개 (독립성, 동질성 검사)

- eg. 생년월일, 스포츠경기 ; 둘은 독립이 아니다.
    성별, 종교 ; 둘은 독립이다.

<실습:교육수준과 흡연율 간의 관련성 분석> - smoke.csv
'''
# 가설
# 귀무가설 H1 : 교육수준과 흡연율은 연관성이 없다.
# 대립가설 H1 : 교육수준과 흡연율은 연관성이 있다.

import pandas as pd
import scipy.stats as stats


data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/smoke.csv')
# print(data.head(3))
# print(data['education'].unique()) # [1 2 3]
# print(data['smoking'].unique()) # [1 2 3] 더미화되어 있다. 숫자에 의미 부여

# 학력별 흡연 인원수를 위한 교차표
ctab = pd.crosstab(index=data['education'], columns=data['smoking']) # edu 독립변수, smoking 종속변수
# print(ctab)
# smoking     1   2   3
# education
# 1          51  92  68
# 2          22  21   9
# 3          43  28  21

## 비율로 보는 법
# ctab = pd.crosstab(index=data['education'], columns=data['smoking'],
#                    normalize=True) # edu 독립변수, smoking 종속변수
# print(round(ctab,4))
# smoking         1       2       3
# education
# 1          0.1437  0.2592  0.1915
# 2          0.0620  0.0592  0.0254
# 3          0.1211  0.0789  0.0592

ctab.index = ['대학원졸','대졸','고졸']
ctab.columns = ['과흡연','보통','노담']
# print(ctab)
#       과흡연  보통  노담
# 대학원졸   51  92   68
# 대졸      22  21    9
# 고졸      43  28   21

chi2, pvalue, dof, _ = stats.chi2_contingency(ctab)
# print(f'chi2 : {chi2:.4f}\npvalue : {pvalue:.6f}\ndof : {dof}')
# chi2 : 18.9109
# pvalue : 0.000818
# dof : 4

msg = "test statics : {}, pvalue : {}, df : {}"
# print(msg.format(chi2, pvalue, dof))

# 결과 : 유의확률 p-value(0.000818) < 유의수준 α(0.05) 이므로 귀무가설을 기각하고, 대립가설을 채택한다.
# 따라서 교육수준과 흡연율 간의 연관성이 있다. 
# ... 어디서 자료를 취했어... 어떤 환경에서 분석했어... 어떻게 자료 가공했어... 어떤 조치를 취해야해...

# p-value가 작아지면 카이제곱이 커짐. 둘 관계는 tradeoff 반비례 관계이다

# -----------------------

'''
음료 종류와 성별 간의 선호도 차이 검정
- 2가지 >> 이원카이제곱
- 남성과 여성의 음료 선호는 서로 관련이 있을까?
'''
# 가설
# 귀무가설 H0 : 성별과 음료 선호는 관련이 있다 (성별에 따라 선호가 같다)
# 대립가설 H1 : 성별과 음료 선호는 관련이 있다 (성별에 따라 선호가 다르다)

data = pd.DataFrame({
    '게토레이':[20, 30],
    '포카리':[10,20],
    '비타500':[10,30],
}, index=['남성','여성'])
print(data)

chi2, pvalue, dof, expected = stats.chi2_contingency(data) # crosstable 모양을 넣어야 한다
print(f'카이제곱 chi2 : {chi2:.4f}\n' # 2.2500
      f'유의확률 pvalue : {pvalue:.6f}\n' # 0.324652
      f'자유도 dof : {dof}\n' # 2
      f'expected :\n{expected}')

# 결론 : 유의확률 p-value(0.324652) > 유의수준 α(0.05) 이므로 대립가설 기각한다.

'''
시각화 : heatmap
- 히트맵은 색상을 활용해 값은 분포를 보여주는 그래프
- 히스토그램이 하나의 변수에 대한 강도(높이)를 활용할 수 있다면,
- 컬러맵은 색상을 활용해 두개의 기준(x축 + y축)에 따른 강도(색상)을 보여준다고 생각하면 된다.
'''
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', family='applegothic')
plt.rcParams['axes.unicode_minus'] = False

sns.heatmap(data, annot=True, fmt='d', cmap='Blues') # annot=True : 숫자 출력
plt.title('성별에 따른 음료 선호')
plt.xlabel('음료')
plt.ylabel('성별')
plt.show()
plt.close()