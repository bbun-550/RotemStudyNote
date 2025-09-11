'''
## 앙상블 Ensemble
- 하나의 샘플 데이터를 여러 개의 분류기를 통해 다수의 학습모델을 만들어 학습시키고 
    학습 결과를 결합하므로써 과적합을 방지하고 정확도를 높이는 학습 기법이다.
3개 할 예정.
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier  # 여러 분류기의 예측을 결합하는 앙상블(투표) 분류기
from sklearn.linear_model import LogisticRegression  # 선형 결정경계를 학습하는 로지스틱 회귀 분류기
from sklearn.neighbors import KNeighborsClassifier  # 가까운 k개의 이웃을 기반으로 분류하는 KNN 분류기
from sklearn.tree import DecisionTreeClassifier  # 규칙을 트리 구조로 학습하는 결정트리 분류기
from collections import Counter  # 레이블 분포 등 항목의 개수를 세는 유틸리티

# 암 데이터로 암 발병 유무 확인
cancer = load_breast_cancer()
x, y = cancer.data, cancer.target
print(x[:2], y[:2], np.unique(y)) # 0 : 악성 malignant, 1 : 양성 benign

# 0과 1의 비율 확인
counter = Counter(y)
total = sum(counter.values())
for cls, cnt in counter.items():
    print(f"class{cls} : {cnt}개 ({cnt/total * 100:.2f}%)")


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=12, stratify=y)
# stratify=y : 레이블 분포가 train / test 고르게 유지하도록 층화 샘플링
# 불균형 데이터에서 모델 평가가 왜곡되지 않도록 한다.

# 분포 확인
y_li = y.tolist()
ytr_li = y_train.tolist()
yte_li = y_test.tolist()
print(f'전체 분포 : {Counter(y_li)}')
print(f'train 분포 : {Counter(ytr_li)}')
print(f'test 분포 : {Counter(yte_li)}')
'''
전체 분포 : Counter({1: 357, 0: 212})
train 분포 : Counter({1: 285, 0: 170})
test 분포 : Counter({1: 72, 0: 42})
'''

# 개별 모델 생성 - scaling 표준화
# make_pipline으로 전처리와 모델을 일체형으로 관리
logi = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver='lbfgs', max_iter=1000, random_state=12)
    )
knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5) # 원 안에 데이터 몇 개 넣을거야?
)
tree = DecisionTreeClassifier(max_depth=5, random_state=12)

# Ensemble 만들 차례 - 개별 모델 합치는 과정
voting = VotingClassifier(
    estimators=[('LR',logi),('KNN',knn),('DT',tree)],
    voting='soft'
)

# 개별 모델 성능
'''
for clf in [logi, knn, tree]:
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    print(f'{clf.__class__.__name__} 정확도 : {accuracy_score(y_test, pred):.4f}')
# Pipeline 정확도 : 0.9912 logi
# Pipeline 정확도 : 0.9737 knn
# DecisionTreeClassifier 정확도 : 0.8772
'''
name_models = [('LR',logi),('KNN',knn),('DT',tree)]
for name, clf in name_models:
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print(f'{name} 정확도 : {accuracy_score(y_test, pred):.4f}')
# LR 정확도 : 0.9912
# KNN 정확도 : 0.9737
# DT 정확도 : 0.8772

voting.fit(x_train, y_train)
vpred = voting.predict(x_test)
print(f'voting 분류기 정확도 : {accuracy_score(y_test, vpred):.4f}')
# voting 분류기 정확도 : 0.9649
# 위에 세 개 평균 값이 voting값

# 옵션 : 교차 검증으로 안정성 확인
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12) # kfold 사용해도 무방
cv_score = cross_val_score(voting, x, y, cv=cv, scoring='accuracy')
print(f'voting 5겹 cv 평균 : {cv_score.mean():.4f} (±{cv_score.std():.4f})')

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
print(classification_report(y_test, vpred, digits=4)) # 실제값, 예측값
print(confusion_matrix(y_test, vpred))
print(roc_auc_score(y_test, voting.predict_proba(x_test)[:,1])) # 0.994047619047619

# GridSearchCV로 최적의 파라미터 찾기
from sklearn.model_selection import GridSearchCV
param_grid = {
    'LR__logisticregression__C':[0.1, 1.0, 10.0],
    'KNN__kneighborsclassifier__n_neighbors':[3,5,7],
    'DT__max_depth':[3,5,7],
}

gs = GridSearchCV(voting, param_grid, cv=cv, scoring='accuracy')
gs.fit(x_train, y_train)
print(f'best params : {gs.best_params_}')
print(f'best cv accuracy : {gs.best_score_}')

best_voting = gs.best_estimator_ # 최적의 모델
print(f'test accuracy(best) : {accuracy_score(y_test, best_voting.predict(x_test))}')
print(f'roc_auc_score(best) : {roc_auc_score(y_test, best_voting.predict_proba(x_test)[:,1])}')
# best params : {'DT__max_depth': 5, 'KNN__kneighborsclassifier__n_neighbors': 5, 'LR__logisticregression__C': 1.0}
# best cv accuracy : 0.9714285714285715
# test accuracy(best) : 0.9649122807017544
# roc_auc_score(best) : 0.994047619047619
