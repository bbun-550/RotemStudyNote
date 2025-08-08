'''
자전거 공유 시스템 관련 파일로 시각화
(워싱턴DC 자료)
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='Malgun Gothic')  # 윈도우: 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False     # 마이너스(-) 깨짐 방지

plt.style.use('ggplot') # R ggplot을 python에서 구현

# 데이터 수집 후 가공(EDA) - train.csv
train = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv', 
                    parse_dates=['datetime'])
# print(train.shape) # (10886, 12)
# print(train.columns)
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')

# train.info() # datatime 월,일,시간 별로 나누고 싶음
# RangeIndex: 10886 entries, 0 to 10885
# Data columns (total 12 columns):
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   datetime    10886 non-null  object > datetime64[ns] # parse_dates=['datetime']로 바꿔줌
#  1   season      10886 non-null  int64
#  2   holiday     10886 non-null  int64
#  3   workingday  10886 non-null  int64
#  4   weather     10886 non-null  int64
#  5   temp        10886 non-null  float64
#  6   atemp       10886 non-null  float64
#  7   humidity    10886 non-null  int64
#  8   windspeed   10886 non-null  float64
#  9   casual      10886 non-null  int64
#  10  registered  10886 non-null  int64
#  11  count       10886 non-null  int64
# dtypes: float64(3), int64(8), object(1)

# pd.set_option('display.max_columns', 500) # 생략된거 볼 수 있게 해줌
# print(train.temp.describe())

# print(train.isnull().sum()) # null값 확인
# null이 포함된 열 확인용 시각화 모듈
# - 모듈 설치 pip install missingno : 결측치 시각화 모듈

# import missingno as msno
# msno.matrix(train, figsize=(12,5))
# plt.show()
# msno.bar(train,figsize=(12,5)) # 막대 그래프
# plt.show()

# 연월일시 데이터로 자전거 대여량 시각화
train['year'] = train['datetime'].dt.year # 연월일시분초 칼럼 생성
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second
# print(train.columns)
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count',
#        'year', 'month', 'day', 'hour', 'minute', 'second'],
#       dtype='object')

'''
figure,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=1, ncols=4) # figure 지정
figure.set_size_inches(15,5)
sns.barplot(data=train, x='year', y='count', ax=ax1)
sns.barplot(data=train, x='month', y='count', ax=ax2)
sns.barplot(data=train, x='day', y='count', ax=ax3)
sns.barplot(data=train, x='hour', y='count', ax=ax4)
ax1.set(ylabel='건수', title='연도별 대여량')
ax2.set(ylabel='건수', title='월별 대여량')
ax3.set(ylabel='건수', title='일별 대여량')
ax4.set(ylabel='건수', title='시별 대여량')
plt.show()
'''

# Boxplot으로 시각화 - 대여량 - 계절별, 시간별 근무일 여부에 따른 대여량
figure,(ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(15,5)
sns.boxplot(data=train, x='season', y='count', ax=ax1)
sns.boxplot(data=train, x='hour', y='count', ax=ax2)
sns.boxplot(data=train, x='workingday', y='count', ax=ax3)

ax1.set(ylabel='대여량', title='계절별 대여량')
ax2.set(ylabel='대여량', title='시간별 대여량')
ax3.set(ylabel='대여량', title='근무일별 대여량')
plt.show()

# -----강사님-----
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12,10)

sns.boxplot(data=train,y='count',orient='v', ax=axes[0][0]) # orient?
sns.boxplot(data=train,y='count',orient='v',x='season', ax=axes[0][1])
sns.boxplot(data=train,y='count',orient='v',x='hour', ax=axes[1][0])
sns.boxplot(data=train,y='count',orient='v',x='workingday', ax=axes[1][1])

axes[0][0].set(ylabel='건수', title='대여량')
axes[0][1].set(xlabel='계절별', title='대여량')
axes[1][0].set(xlabel='시간별', title='대여량')
axes[1][1].set(xlabel='근무일', title='대여량')
plt.show()
