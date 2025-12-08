'''
[로지스틱 분류분석 문제1]
소득 수준에 따른 외식 성향을 나타내고 있다. 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라.
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score

# 데이터 불러오기
data = pd.read_csv('/Users/bunny/Documents/git_practice/python_analysis/analysis05/ex.csv')
# print(data.head(5))
# data.info()
# print(f'소득 평균 : {np.mean(data['소득수준']):.2f}') # 소득 평균 : 50.71

# 불필요한 칼럼 제거
data = data[data['요일'].isin(['토', '일'])].reset_index(drop=True)
data = data.drop(['요일'],axis=1)

# train/test 분리
train, test = train_test_split(data, test_size=0.3, random_state=33)
# print(train.shape, test.shape)

# 분류모델
model = smf.glm(formula='외식유무 ~ 소득수준', data=train, family=sm.families.Binomial()).fit()
print(model.summary()) # pvalue 0.010 < 0.05
print(f'예측값 : {np.round(model.predict(test).values)}')
print(f'실제값 : {test['외식유무'].values}')
# 예측값 : [0 0 0 0 1 1 0]
# 실제값 : [0 0 0 1 1 1 0]

# 정확도 평가
print(f'모델 정확도 : {accuracy_score(test['외식유무'], np.round(model.predict(test)))}')
# 모델 정확도 : 0.8571428571428571

# confusion matrix로 정확도 평가
'''
model1 = smf.logit(formula='외식유무 ~ 소득수준', data=train).fit()
conf_tab = model1.pred_table()
print(conf_tab)
print(f'conf_tab 정확도 : {(conf_tab[0][0]+conf_tab[1][1])/conf_tab.sum()}')
# conf_tab 정확도 : 0.9047619047619048
'''


try:
    num = int(input('키보드로 소득 수준(양의 정수)을 입력 :').strip())
    new_df = pd.DataFrame({'소득수준':[num]})
    pred = np.round(model.predict(new_df)[0])
    print(f'결과 : {"주말에 외식한다" if pred==1 else "주말에 외식하지 않는다"}')
except ValueError:
    print("입력 오류")