'''
[Randomforest 문제2]
중환자 치료실에 입원 치료 받은 환자 200명의 생사 여부에 관련된 자료다.
종속변수 STA(환자 생사 여부)에 영향을 주는 주요 변수들을 이용해 검정 후에 해석하시오. 
모델 생성 후 입력자료 및 출력결과는 Django를 사용하시오.

예제 파일 : https://github.com/pykwon  ==>  patient.csv

<변수설명>
  STA : 환자 생사 여부 (0:생존, 1:사망)
  AGE : 나이
  SEX : 성별
  RACE : 인종
  SER : 중환자 치료실에서 받은 치료
  CAN : 암 존재 여부
  INF : 중환자 치료실에서의 감염 여부
  CPR : 중환자 치료실 도착 전 CPR여부
  HRA : 중환자 치료실에서의 심박수
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/patient.csv')

# df.info()
# print(df.head())
# print(df.isnull().sum())

x=df.drop(['ID','STA'], axis=1)
y=df['STA']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

logimodel = LogisticRegression(solver='lbfgs', max_iter=500).fit(x_train, y_train)
dtmodel = DecisionTreeClassifier().fit(x_train, y_train)
rfmodel = RandomForestClassifier().fit(x_train, y_train)

logipred = logimodel.predict(x_test)
print(f'logimodel acc : {accuracy_score(y_test,logipred):.5f}')

dtpred = dtmodel.predict(x_test)
print(f'dtmodel acc : {accuracy_score(y_test,dtpred):.5f}')

rfpred = rfmodel.predict(x_test)
print(f'rfmodel acc : {accuracy_score(y_test,rfpred):.5f}')
