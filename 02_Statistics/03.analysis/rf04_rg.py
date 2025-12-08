# RandomForestRegressor : 정량적(분류가 아니라 수치 예측) 예측 모델
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

housing = fetch_california_housing(as_frame=True) # dataframe 형태 반환 옵션, 그냥 출력하면 배열로 출력됨
# print(housing.data[:2], housing.target[:2])
# print(housing.feature_names)
df = housing.frame # as_frame=True 해서 가능
print(df.head(2)) # 전부 연속형 데이터

# feature 와 label로 분리
x = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# 랜덤포레스트 ; n_estimators 생성할 트리 개수(기본 500개) / 
# n_jobs 모델 학습 및 예측에 사용되는 병렬 작업의 수 GPU 사용해라(1 : 모든 프로세서 사용 병렬처리)
# 
rfmodel = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rfmodel.fit(x_train, y_train)

y_pred = rfmodel.predict(x_test)
print(f'MSE : {mean_squared_error(y_test, y_pred):.3f}') # 0에 근사하면 유의한 결과
print(f'R2 : {r2_score(y_test, y_pred):.3f}\n')

print('독립변수 중요도 순위표')
importance = rfmodel.feature_importances_
indices = np.argsort(importance)[::-1] # 내림차순

ranking = pd.DataFrame({
    'Feature':x.columns[indices],
    'Importance':importance[indices]
})
print(ranking)

# 간단한 튜닝으로 최적의 파라마터 찾기
# GridSearchCV : 정확하게 최적값 찾기 적당, 파라미터가 많으면 계산량 폭발적 증가
from sklearn.model_selection import RandomizedSearchCV # 연속형 데이터 다룰 때 효과적

param_list = {
    'n_estimators':[200,400,600], # 트리 개수
    'max_depth':[None,10,20,30], # 
    'min_samples_leaf':[1,2,4], # 리프 노드 최소 샘플 수
    'min_samples_split':[2,5,10], # 노드 분할 최소 샘플 수
    'max_features':[None, 'sqrt','log2', 1.0, 0.8, 0.6] # 최대 특성수
}
search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42), # 기준 모델
    param_distributions=param_list,
    n_iter=20, # random search 탐색 횟수
    cv=3 , # cross validation 검증을 위한 분할 검증 횟수 ;3겹으로 교차검정 
    scoring='r2', # 오차 평가방법
    random_state=42
)

search.fit(x_train, y_train)

print(f'best params : {search.best_params_}')
# 최적 모델 추출
best_model = search.best_estimator_
print(f'best cv r2(교차검증 평균 결정계수) : {search.best_score_}')
print(f'best_model 결정계수 : {r2_score(y_test, best_model.predict(x_test))}')