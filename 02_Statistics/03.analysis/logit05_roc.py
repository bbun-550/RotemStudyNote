## 분류모델 성능평가
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression # 분류모델(정성적 모델)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=123) # 샘플수 100개, 독립변수 2개
print(x[:3]) # 2차원
print(y[:3]) # 1차원
'''
# 굳이 산포도 시각화
import matplotlib.pyplot as plt
plt.scatter(x[:,0], x[:,1])
plt.show()
'''

# 모델
model = LogisticRegression().fit(x,y)
yhat = model.predict(x)
print(f'예측값 : {yhat[:3]}')

# 결정함수(판별함수) : 불확실성 추정함수 => 판별 경계선 설정하기 위한 샘플 자료 얻기
f_value = model.decision_function(x)
print(f'f_value : {f_value[:10]}') # 정상 출력되는지 확인 

# 실제값, 예측값을 df에 담아보자
df = pd.DataFrame(np.vstack([f_value, yhat, y]).T, columns=['f','yhat','y'])
# print(df.head(3))

# confusion matrix 혼동행렬
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, yhat)) # 실제값,예측값
# [[44(TP)  4(FN)]
#  [ 8(FP) 44(TN)]]
accuracy = (44+44) / 100
recall = 44 / (44 + 4)
precision = 44 / (44 + 8)
specificity = 44 / (8 + 44) # TN / FP + TN
fallout = 8 / (8 + 44) # 위양성률 fallout FP/(FP + TN)
print(f'정확도 : {accuracy}')
print(f'재현률 : {recall}')
print(f'정밀도 : {precision}')
print(f'특이도 : {specificity}')
print(f'위양성률1 : {fallout}')
print(f'위양성률2 : {1 - specificity}')
'''
정리
- TPR은 1에 근사하면 좋고, FPR은 0에 근사하면 좋다
'''

# 손으로 계산하지 말고 메소드로 구하자
from sklearn import metrics
ac_score = metrics.accuracy_score(y, yhat)
print(f'정확도(메소드) : {ac_score}')
cl_rep = metrics.classification_report(y, yhat)
'''
print(cl_rep) # precision, recall도 보여준다.
              precision    recall  f1-score   support

           0       0.85      0.92      0.88        48
           1       0.92      0.85      0.88        52

    accuracy                           0.88       100
   macro avg       0.88      0.88      0.88       100 # 가중치 평균
weighted avg       0.88      0.88      0.88       100 # support 가중치 평균
'''
# 시각화 준비
fpr, tpr, thresholds = metrics.roc_curve(y, model.decision_function(x)) # 실제값,
'''
print(f'fpr : {fpr}')
print(f'tpr : {tpr}')
print(f'분류임계결정값 : {thresholds}') # 이 값들이 x축, y축으로 들어간다. 분류 임계값은 선으로
'''

# ROC 본격 시각화
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

# AUC Area Under the Curve : ROC curve의 면적
# - 1에 가까울수록 좋은 모델이라고 본다.
print(f'AUC : {metrics.auc(fpr, tpr)}') # 0.9547