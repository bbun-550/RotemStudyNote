'''
https://www.kaggle.com/c/bike-sharing-demand/data 에서 train.csv를 다운받아 bike_dataset.csv 으로 파일명을 변경한다. 
이 데이터는 어느 지역의 2011년 1월 ~ 2012년 12월 까지 날짜/시간. 기온, 습도, 풍속 등의 정보를 바탕으로 1시간 간격의 자전거 대여횟수가 기록되어 있다.
1. train / test로 분류 한 후 
2. 대여횟수에 중요도가 높은 칼럼을 판단하여 feature를 선택한 후, 
3. 대여횟수에 대한 회귀 예측(RandomForestRegressor)을 하시오.
(배점:10)
칼럼 정보 :
  'datetime', 'season'(사계절:1,2,3,4),  'holiday'(공휴일(1)과 평일(0)), 'workingday'(근무일(1)과 비근무일(0)),
  'weather'(4종류:Clear(1), Mist(2), Snow or Rain(3), Heavy Rain(4)),
  'temp'(섭씨온도), 'atemp'(체감온도), 'humidity'(습도), 'windspeed'(풍속),
  'casual'(비회원 대여량), 'registered'(회원 대여량), 'count'(총대여량)
참고 : casual + registered 가 count 임.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score

data = pd.read_csv('bike_dataset.csv', parse_dates=['datetime'])
# print(data.head())
# print(data.columns)
'''
['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
       'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
'''

# data.info()
'''
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   datetime    10886 non-null  object 
 1   season      10886 non-null  int64  
 2   holiday     10886 non-null  int64  
 3   workingday  10886 non-null  int64  
 4   weather     10886 non-null  int64  
 5   temp        10886 non-null  float64
 6   atemp       10886 non-null  float64
 7   humidity    10886 non-null  int64  
 8   windspeed   10886 non-null  float64
 9   casual      10886 non-null  int64  
 10  registered  10886 non-null  int64  
 11  count       10886 non-null  int64 
'''
# msno.matrix(data, figsize=(12,5))
# plt.show() # 결측치 시각화 : 결측치 없음

data['year'] = data['datetime'].dt.year # 연월일시분초 칼럼 생성
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['hour'] = data['datetime'].dt.hour
data['minute'] = data['datetime'].dt.minute
data['second'] = data['datetime'].dt.second

data = data.query('year < 2013')

data = data.drop(['casual','registered'], axis=1)
print(data.head())
print(data.year.unique())

corr = data.corr(method='pearson').round(2)
# print(corr)

# 상관계수 높은 변수 : hour, humidity, atemp, temp
feature = data[['hour','humidity','atemp','temp']]
label = data['count']

# train/test split
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=0)

# 모델
model = RandomForestRegressor(n_estimators=500, random_state=12, n_jobs=-1)
model.fit(x_train, y_train)

pred = model.predict(x_test)
print(f'예측값 : {pred[:5]}')
print(f'실제값 : {y_test[:5].values}')
print(f'R2 : {r2_score(y_test, pred):.3f}\n')



