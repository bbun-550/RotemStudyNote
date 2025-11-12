import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB # 연속 데이터일 때 사용. 이진 분류 일 떄는 베르누이. multinomial
from sklearn.metrics import accuracy_score
from sklearn import metrics

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv')
# df.info()
# print(df['RainTomorrow'].unique()) # ['Yes' 'No'] -> 더미화해줘야 한다.
x = df[['MinTemp','MaxTemp','Rainfall']]
label = df['RainTomorrow'].map({'Yes':1, 'No':0})
# print(x[:3],'\n',label[:3])

# train/test
train_x, test_x, train_y, test_y = train_test_split(x,label, test_size=0.2, random_state=0)

# 모델 생성 및 학습
gmodel = GaussianNB()
gmodel.fit(train_x, train_y)
pred = gmodel.predict(test_x)
print(f'예측값 : {pred[:10]}')
print(f'실제값 : {test_y[:10].values}')
acc = sum(test_y == pred) / len(pred)
print(f'정확도1 : {acc:.4f}')
print(f'정확도2 : {accuracy_score(test_y, pred):.4f}') # acc : 0.7703

 