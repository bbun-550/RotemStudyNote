'''
회귀분석 문제 1) scipy.stats.linregress() <= 꼭 하기 : 심심하면 해보기 => statsmodels ols(), LinearRegression 사용
나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 하루 평균 시청 시간과 운동량에 대한 데이터는 아래와 같다.
 - 지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
 - 지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
    참고로 결측치는 해당 칼럼의 평균 값을 사용하기로 한다. 이상치가 있는 행은 제거. 운동 10시간 초과는 이상치로 한다.  

구분,지상파,종편,운동
1,0.9,0.7,4.2,
2,1.2,1.0,3.8,
3,1.2,1.3,3.5,
4,1.9,2.0,4.0,
5,3.3,3.9,2.5,
6,4.1,3.9,2.0,
7,5.8,4.1,1.3,
8,2.8,2.1,2.4,
9,3.8,3.1,1.3,
10,4.8,3.1,35.0,
11,NaN,3.5,4.0,
12,0.9,0.7,4.2,
13,3.0,2.0,1.8,
14,2.2,1.5,3.5,
15,2.0,2.0,3.5
'''

'''
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.rc('font',family='applegothic')

raw = [
    1,0.9,0.7,4.2,
    2,1.2,1.0,3.8,
    3,1.2,1.3,3.5,
    4,1.9,2.0,4.0,
    5,3.3,3.9,2.5,
    6,4.1,3.9,2.0,
    7,5.8,4.1,1.3,
    8,2.8,2.1,2.4,
    9,3.8,3.1,1.3,
    10,4.8,3.1,35.0,
    11,np.nan,3.5,4.0,
    12,0.9,0.7,4.2,
    13,3.0,2.0,1.8,
    14,2.2,1.5,3.5,
    15,2.0,2.0,3.5
]
raw_array = np.array(raw).reshape(-1,4)
data1 = pd.DataFrame(raw_array, columns=['구분','지상파','종편','운동']).set_index('구분')
jishang = np.mean(data1['지상파']).round(2)
data1 = data1.fillna(jishang)
data1 = data1.drop(data1[data1['운동'] >= 10].index, axis = 0).reset_index(drop=True)
# print(data1)

x1 = data1.지상파
y1 = data1.운동

model1 = stats.linregress(x1,y1)
# print(f'기울기 : {model1.slope:.4f}')
# print(f'절편 : {model1.intercept:.4f}')
# print(f'상관계수 r : {model1.rvalue:.4f}')
# print(f'p-value : {model1.pvalue:.4f}')
# print(f'표준오차 : {model1.stderr:.4f}')
# 기울기 : 1.6678
# 절편 : 0.6994
# 상관계수 r : -0.8654 지상파 시청시간이 많을수록 운동시간이 줄어드는 경향이 있다.
# R2 = -0.8654 ** 2 즉, R2 = 0.74891716 독립변수가 종속변수를 74% 정도 설명하고 있다. 즉, 지상파 시청시간은 운동시간과 74% 관련이 있다.
# p-value : 0.0001 < 0.05 이므로 통계적으로 유의하다. 지상파 시청시간과 운동시간은 연관이 있다.
# 표준오차 : 1.5685

x2 = data1.지상파
y2 = data1.종편
model2 = stats.linregress(x2,y2)
# print(f'기울기 : {model2.slope:.4f}')
# print(f'절편 : {model2.intercept:.4f}')
# print(f'R2 결정계수 : {model2.rvalue:.4f}')
# print(f'p-value : {model2.pvalue:.4f}')
# print(f'표준오차 : {model2.stderr:.4f}')
# 기울기 : 0.7728
# 절편 : 0.2947
# R2 결정계수 : 0.8877 
# p-value : 0.0000 
# 표준오차 : 0.1157

jishangpa_time = int(input('지상파 시청 시간 입력 : '))
print(f'지상파 시청 시간에 따른 운동시간 예측 결과 : {np.polyval([model1.slope, model1.intercept], np.array([jishangpa_time]))}시간')
print(f'지상파 시청 시간에 따른 종편 시청 시간 예측 결과 : {np.polyval([model2.slope, model2.intercept], np.array([jishangpa_time]))}시간')

plt.subplot(1,2,1)
plt.scatter(x1,y1)
plt.plot(x1, model1.slope * x1 + model1.intercept)
plt.xlabel('지상파 시청 시간')
plt.ylabel('운동시간')
r2 = model1.rvalue**2
plt.title(f'지상파-운동 (R²={r2:.2f})')

plt.subplot(1,2,2)
plt.scatter(x2,y2)
plt.plot(x2, model2.slope * x2 + model2.intercept)
plt.xlabel('지상파 시청 시간')
plt.ylabel('종편 시청 시간')
plt.title('지상파 시청 시간 - 종편')
plt.tight_layout()
plt.show()
plt.close()
'''


'''
회귀분석 문제 2) 
testdata에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다. 
이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오.  수학점수를 종속변수로 하자.
  - 국어 점수를 입력하면 수학 점수 예측
  - 국어, 영어 점수를 입력하면 수학 점수 예측
'''
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='applegothic')
import scipy.stats as stats
import numpy as np

raw2 = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv'
data2 = pd.read_csv(raw2)
# print(data2.head(2))

# 데이터 칼럼 확인
# print(data2.columns)

# 이상치 확인
# data2.info()

# 이상치 시각화
# plt.boxplot([data2.국어, data2.영어, data2.수학])
# plt.xticks([1,2,3], ['국어','영어','수학'])
# plt.show()
# plt.close()

# 독립변수
lang1 = data2.국어

# 종속변수
math1 = data2.수학


model1 = stats.linregress(lang1, math1)
# print(f'기울기 : {model1.slope:.4f}')
# print(f'절편 : {model1.intercept:.4f}')
# print(f'상관계수 R : {model1.rvalue:.4f}')
# print(f'p-value : {model1.pvalue:.4f}')
# print(f'표준오차 : {model1.stderr:.4f}')
# print(f'R2 결정계수 : {model1.rvalue**2:.4f}')
# 기울기 : 0.5705
# 절편 : 32.1069
# 상관계수 R : 0.7663 -> 국어 점수와 수학 점수는 양의 상관관계를 갖는다. 국어 점수가 높으면 수학 점수도 높다.
# p-value : 0.0001 < 0.05 이므로 통계적으로 유의하다.
# 표준오차 : 0.1128
# R2 결정계수 : 0.5872 -> 독립변수가 종속변수를 58% 정도 설명하고 있다. 국어 점수와 수학 점수는 58% 관련이 있다.

# lang_score = int(input('국어 점수 입력 : '))
# print(f'수학 점수 예측 결과 : {np.polyval([model1.slope, model1.intercept], np.array([lang_score]))}점')
# 국어 점수 입력 : 70
# 수학 점수 예측 결과 : [72.0454057]점


# 독립변수
lang2 = data2.국어
eng2 = data2.영어

# 종속변수
math2 = data2.수학


