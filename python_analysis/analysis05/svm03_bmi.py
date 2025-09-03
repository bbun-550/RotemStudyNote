# BMI 식을 이용해 데이터 만들기
# 비만도 계산은 몸무게를 키의 제곱으로 나눈 것
# eg. 키 175, 몸무게 68 => 68 / ((170 / 100) * (170 / 100))

'''
# 빅데이터 만들기
import random
random.seed(12)
def calc_bmi(h,w):
    bmi = w / (h/100)**2
    if bmi < 18.5:return 'thin'
    if bmi < 25:return 'normal'
    return 'fat'
print(calc_bmi(170,88))
fp = open('bmi.csv', 'w')
fp.write('height,weight,label\n') # 제목

cnt = {'thin':0, 'normal':0, 'fat':0}
for i in range(50000):
    h = random.randint(150,200)
    w = random.randint(35, 100)
    label = calc_bmi(h,w)
    cnt[label] += 1
    fp.write('{0},{1},{2}\n'.format(h,w,label))

fp.close()
'''
# SVM으로 분류모델 만들기
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

tbl = pd.read_csv('bmi.csv')
# print(tbl.head(), tbl.shape) # (50000, 3)

label = tbl['label']

# 정규화
w = tbl['weight'] / 100
print(w[:3].values)
h = tbl['height'] / 200
print(h[:3].values)

wh = pd.concat([w,h], axis=1)
print(wh.head(3), wh.shape) # (50000, 2)

# label 정수화(더미화) : svm은 사실 더미화하지 않아도 알아서 처리함
label = label.map({'thin':0, 'normal':1, 'fat':2})
print(label[:3])

x_train, x_test, y_train, y_test = train_test_split(wh,label, test_size=0.3, random_state=1)

# 모델
model = svm.SVC(C=0.01, kernel='rbf', random_state=1).fit(x_train, y_train) # C : 규제강도. 숫자 값(마진폭)이 커지면 과적합할 가능성이 높다.

pred = model.predict(x_test)
print(f'예측값 : {pred[:10]}')
print(f'실제값 : {y_test[:10].values}')

ac_score = metrics.accuracy_score(y_test, pred)
print(f'정확도 : {ac_score}') # 정확도 : 0.9705333333333334

tbl2 = pd.read_csv('bmi.csv', index_col=2)
def scatterFunc(lbl, color):
    b = tbl2.loc[lbl]
    plt.scatter(b['weight'],b['height'], c=color, label=lbl)

# scatterFunc('fat', 'red')
# scatterFunc('normal', 'yellow')
# scatterFunc('thin', 'blue')
# plt.legend()
# plt.show()
# plt.close()

# 새로운 값으로 예측
newData = pd.DataFrame({'weight':[19,90], 'height':[100,168]})
# 학습시킬 때 정규화했으므로, 새로운 데이터도 정규화 시켜서 predict에 넣어야 한다.
newData['weight'] = newData['weight']/100
newData['height'] = newData['height']/100
new_pred = model.predict(newData)
print(f'새로운 데이터에 대한 bmi는 {new_pred}')
