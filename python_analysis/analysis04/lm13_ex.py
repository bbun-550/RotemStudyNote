'''
회귀분석 문제 5) 
- Kaggle 지원 dataset으로 회귀분석 모델(LinearRegression)을 작성하시오.
- testdata 폴더 : Consumo_cerveja.csv
- Beer Consumption - Sao Paulo : 브라질 상파울루 지역 대학생 그룹파티에서 맥주 소모량 dataset
- feature : Temperatura Media (C) : 평균 기온(C)

            Precipitacao (mm) : 강수(mm)
- label : Consumo de cerveja (litros) - 맥주 소비량(리터) 를 예측하시오
- 조건 : NaN이 있는 경우 삭제!
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Consumo_cerveja.csv')
print(data.head(3))
# data.info()
#  #   Column                       Non-Null Count  Dtype
# ---  ------                       --------------  -----
#  0   Data                         365 non-null    object
#  1   Temperatura Media (C)        365 non-null    object
#  2   Temperatura Minima (C)       365 non-null    object
#  3   Temperatura Maxima (C)       365 non-null    object
#  4   Precipitacao (mm)            365 non-null    object
#  5   Final de Semana              365 non-null    float64
#  6   Consumo de cerveja (litros)  365 non-null    float64

# print(data.columns)
# ['Data', 'Temperatura Media (C)', 'Temperatura Minima (C)',
#        'Temperatura Maxima (C)', 'Precipitacao (mm)', 'Final de Semana',
#        'Consumo de cerveja (litros)']
data = data.dropna()
data['Temperatura Maxima (C)'] = data['Temperatura Maxima (C)'].replace(',','.', regex=True)
data['Precipitacao (mm)'] = data['Precipitacao (mm)'].replace(',','.', regex=True)
data['Temperatura Maxima (C)'] = pd.to_numeric(data['Temperatura Maxima (C)'], errors='coerce')
data['Precipitacao (mm)'] = pd.to_numeric(data['Precipitacao (mm)'], errors='coerce')
# print(data['Temperatura Maxima (C)'])
# data.info()

# 데이터 추출
x1 = data[['Temperatura Maxima (C)']].values
x2 = data[['Precipitacao (mm)']].values
y = data['Consumo de cerveja (litros)'].values

# 상관계수 확인
print('온도 ~ 맥주 상관계수 : %.4f'%(np.corrcoef(x1.flatten(),y)[0,1])) # 0.6427
print('강수 ~ 맥주 상관계수 : %.4f'%(np.corrcoef(x2.flatten(),y)[0,1])) # -0.1938
# 강수와 맥주의 상관관계는 유의하지 않다.

# 모델
print('---------<온도로 맥주 소비량 예측 결과>--------')
lmodel1 = LinearRegression().fit(x1,y)
print(f'lmodel1 slope : {lmodel1.coef_[0]:.4f}') # 기울기 : 0.6548
print(f'lmodel1 intercept : {lmodel1.intercept_:.4f}') # 절편 : 7.9749

predict1 = lmodel1.predict(x1)
print(f'온도로 예측한 맥주 : {predict1[0]:.4f}')
print(f'r2_score(결정계수): {r2_score(y, predict1):.4f}') # 0.4130 => 41.30% 설명해주고 있다.
print(f'MSE(평균제곱오차): {mean_squared_error(y, predict1):.4f}') # 11.3282

print('---------<강수량으로 맥주 소비량 예측 결과>--------')
lmodel2 = LinearRegression().fit(x2,y)
print(f'lmodel2 slope : {lmodel2.coef_[0]:.4f}') # 기울기 : -0.0686
print(f'lmodel2 intercept : {lmodel2.intercept_:.4f}') # 절편 : 25.7581

predict2 = lmodel2.predict(x2)
print(f'강수량으로 예측한 맥주 : {predict2[0]:.4f}')
print(f'r2_score(결정계수): {r2_score(y, predict2):.4f}') # 0.0376 => 3.76% 설명해주고 있다.
print(f'MSE(평균제곱오차): {mean_squared_error(y, predict2):.4f}') # 18.5747

