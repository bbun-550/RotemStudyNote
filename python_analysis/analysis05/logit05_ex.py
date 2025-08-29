from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression # 분류모델(정성적 모델)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

'''
[로지스틱 분류분석 문제2] 
게임, TV 시청 데이터로 안경 착용 유무를 분류하시오.
안경 : 값0(착용X), 값1(착용O)
예제 파일 : https://github.com/pykwon  ==>  bodycheck.csv
새로운 데이터(키보드로 입력)로 분류 확인. 스케일링X
'''
'''
data1 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/bodycheck.csv')
# print(data1.head(2)) # ok
data1 = data1[['게임','TV시청','안경유무']]
# print(data1.head(2)) # ok

x1 = data1[['게임','TV시청']]
print(type(x1), x1.shape)
y1 = data1['안경유무']
print(type(y1), y1.shape)

random = [0, 42, 88, 123, 624]
for seed,k in enumerate(random):
    print('------------------------')
    x_train, x_test, y_train, y_test = train_test_split(x1,y1, test_size=0.3, random_state=seed)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    model1 = LogisticRegression().fit(x_train,y_train)
    y_pred1 = model1.predict(x_test)
    print(f'예측값\n{y_pred1}')
    print(f'실제값\n{y_test.values}')

    # 정확도
    accuracy1 = metrics.accuracy_score(y_test, y_pred1)
    print(f'{k}.정확도 : {accuracy1}')
'''

# 시각화
'''
fpr, tpr, thresholds = metrics.roc_curve(y_test, model1.decision_function(x_test))

plt.plot(fpr, tpr, 'o-', label='LogisticRegression')
plt.plot([0,1],[0,1],'k--', label='Random Classifier Line(AUC 0.5)') # 경계선
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()
'''


'''
[로지스틱 분류분석 문제3]
Kaggle.com의 https://www.kaggle.com/truesight/advertisingcsv  file을 사용
얘 사용해도 됨   'testdata/advertisement.csv' 

참여 칼럼 : 
   - Daily Time Spent on Site : 사이트 이용 시간 (분)
   - Age : 나이,
   - Area Income : 지역 소득,
   - Daily Internet Usage :일별 인터넷 사용량(분),
   - Clicked Ad : 광고 클릭 여부 ( 0 : 클릭x , 1 : 클릭o )
광고를 클릭('Clicked on Ad')할 가능성이 높은 사용자 분류.
데이터 간 단위가 큰 경우 표준화 작업을 시도한다.
모델 성능 출력 : 정확도, 정밀도, 재현율, ROC 커브와 AUC 출력
새로운 데이터로 분류 작업을 진행해 본다.

'''
data2 = pd.read_csv(r'C:\Users\acorn\Documents\git_practice\python_analysis\analysis05\advertising.csv')
# print(data2.columns) # ['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Ad Topic Line', 'City', 'Male', 'Country','Timestamp', 'Clicked on Ad']
# print(data2.head(2))
# data2.info()

data2 = data2[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage','Clicked on Ad']]
# print(data2.head())

# 데이터 간 상관관계
independent = ['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage']
dependent = ['Clicked on Ad']
print(np.corrcoef(independent[0]))

x = data2[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage']]
print(type(x), x.shape)
y = data2['Clicked on Ad']
print(type(y), y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=123)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 표준화
scaler = StandardScaler()
x_train_scale = scaler.fit_transform(x_train)
x_test_scale = scaler.transform(x_test)


model = LogisticRegression().fit(x_train_scale,y_train)
y_pred = model.predict(x_test_scale)
print(f'예측값\n{y_pred}')
print(f'실제값\n{y_test.values}')

# 정확도
accuracy = metrics.accuracy_score(y_test, y_pred)
cl_rep = metrics.classification_report(y_test, y_pred)
print(cl_rep)
print(confusion_matrix(y_test, y_pred))
TP,FN,FP,TN = (150,3,4,143)
recall = TP / (TP+FN)
precision = TP / (TP + FP)
fallout = FP/(FP + TN)
print(f'정확도 : {accuracy:.4f}')
print(f'재현률 : {recall:.4f}')
print(f'정밀도 : {precision:.4f}')

# ROC 시각화
fpr, tpr, thresholds = metrics.roc_curve(y_test, model.decision_function(x_test)) # 실제값,


plt.plot(fpr, tpr, 'o-', label='LogisticRegression')
plt.plot([0,1],[0,1],'k--', label='Random Classifier Line(AUC 0.5)') # 경계선 그어준다 / AUC란? Area Under the Curve
plt.plot([fallout],[recall],'ro',ms=10) # 위양성률과 재현율 값 출력
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

'''
'''