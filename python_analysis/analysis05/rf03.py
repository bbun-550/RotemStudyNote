# titanic dataset : LogisticRegression, DecisionTreeClassifier, RandomForestClassifier 성능 비교
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv')
print(df.head())

df.drop(columns=['PassengerId','Name','Ticket'], inplace=True)
print(df.isnull().sum())

# NuLL 처리 : 평균 또는 'N' 으로 변경
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Cabin'].fillna('N', inplace=True)
df['Embarked'].fillna('N', inplace=True)
print(df.isnull().sum())
df.info()

# oject type변환
print(f'Sex : {df.Sex.value_counts()}')
print(f'Cabin : {df.Cabin.value_counts()}')
print(f'Embarked : {df.Embarked.value_counts()}')
df['Cabin'] = df['Cabin'].str[:1]

# 성별이 생존 확률에 어떤 영향을 미쳤는지 확인하기
print(df.groupby(['Sex','Survived'])['Survived'].count())
print(f'여성 생존율 : {233/(81+233) * 100:.2f}%')
print(f'남성 생존율 : {109/(468+109) * 100:.2f}%')

# 시각화
sns.barplot(x='Sex', y='Survived', data=df, errorbar=('ci', 95))
# plt.show()
# plt.close()

# 성별 기준 Pclass별 생존확률
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df)
# plt.show()
# plt.close()

# 나이별 기준으로 생존 확률
def getAgeFunc(age):
    msg =''
    if age <= -1:msg = 'unknown'
    elif age <= 5:msg = 'baby'
    elif age <= 18:msg = 'teenager'
    elif age <= 65:msg = 'adult'
    else: msg='elder'
    return msg

df['Age_category'] = df['Age'].apply(lambda a:getAgeFunc(a))
print(df.head())

sns.barplot(x='Age_category', y='Survived', data=df, hue='Sex',
            order=['unknown','baby','teenager','adult','elder'])
# plt.show()
# plt.close()
del df['Age_category']

# 문자열 자료를 숫자화
from sklearn import preprocessing
def labelIncoder(datas):
    cols = ['Cabin','Sex','Embarked']
    for c in cols:
        lab = preprocessing.LabelEncoder()
        lab = lab.fit(datas[c])
        datas[c] = lab.transform(datas[c])
    return datas

df = labelIncoder(df)
print(df.head(3))
print(df['Cabin'].unique()) # [7 2 4 6 3 0 1 5 8]
print(df['Sex'].unique()) # [1 0]
print(df['Embarked'].unique()) # [3 0 2 1]

# feature/label 분리
feature_df = df.drop(['Survived'], axis='columns')
label_df = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(feature_df, label_df, test_size=0.2, random_state=1)

logimodel = LogisticRegression(solver='lbfgs', max_iter=500).fit(x_train, y_train)
dtmodel = DecisionTreeClassifier().fit(x_train, y_train)
rfmodel = RandomForestClassifier().fit(x_train, y_train)

logipred = logimodel.predict(x_test)
print(f'logimodel acc : {accuracy_score(y_test,logipred):.5f}') # logimodel acc : 0.79888

dtpred = dtmodel.predict(x_test)
print(f'dtmodel acc : {accuracy_score(y_test,dtpred):.5f}') # dtmodel acc : 0.72067

rfpred = rfmodel.predict(x_test)
print(f'rfmodel acc : {accuracy_score(y_test,rfpred):.5f}') # rfmodel acc : 0.75978
