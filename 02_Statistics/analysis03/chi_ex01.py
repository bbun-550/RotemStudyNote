'''
카이제곱 검정

카이제곱 문제1) 부모학력 수준이 자녀의 진학여부와 관련이 있는가?를 가설검정하시오
  예제파일 : cleanDescriptive.csv
  칼럼 중 level - 부모의 학력수준, pass - 자녀의 대학 진학여부
  조건 :  level, pass에 대해 NA가 있는 행은 제외한다.
'''
# 가설
# 귀무 : 부모학력 수준이 자녀의 진학여부와 관련이 없다.
# 대립 : 부모학력 수준이 자녀의 진학여부와 관련이 있다.

import pandas as pd
import scipy.stats as stats

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/cleanDescriptive.csv')
# print(data.head(3))

# 더미 데이터 확인 후 숫자에 의미 부여
# print(data['level'].unique()) # [ 1.  2. nan  3.] level - 부모의 학력수준 : 1 - 고졸, 2 - 대졸, 3 - 대학원졸
# print(data['pass'].unique()) # [ 2.  1. nan] pass - 자녀의 대학 진학여부 : 2 - 합격, 1 - 실패


ndata = data[['level','pass']]
ndata = ndata.dropna().reset_index(drop=True) # 결측치 포함 행 제거 후 인덱스 리셋
# print(ndata.head())

ctab_data = pd.crosstab(index=ndata['level'], columns=ndata['pass'])
ctab_data.index = ['대학원졸','대졸','고졸'] # ['대학원졸','대졸','고졸'] ['고졸','대졸','대학원졸']
ctab_data.columns = ['불합격','합격'] # ['합격','불합격'] ['불합격','합격']
print(ctab_data)

chi2, pvalue, dof, _ = stats.chi2_contingency(ctab_data)
print(f'카이제곱 chi2 : {chi2:.4f}\n'
      f'유의확률 pvalue : {pvalue:.6f}\n'
      f'자유도 dof : {dof}\n')
# 카이제곱 chi2 : 2.7670
# 유의확률 pvalue : 0.250706
# 자유도 dof : 2

# 결론 : p-value(0.250706) > α(0.05) 이므로 귀무 가설 기각하지 않는다.
# 부모학력과 자녀의 진학 여부는 관련이 있다는 주장은 우연히 발생한 것이다.