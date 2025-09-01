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
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression # classifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier

# 암 데이터로 암 발병 유무 확인
cancer = load_breast_cancer()
x, y = cancer.data, cancer.target
print(x[:2], y[:2], np.unique(y)) # 0 : 악성 malignant, 1 : 양성 benign

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=12, stratify=y)
# stratify=y : 레이블 분포가 train / test 고르게 유지하도록 층화 샘플링
# 불균형 데이터에서 모델 평가가 왜곡되지 않도록 한다.

# 분포 확인
from collections import Counter
# print(f'전체 분포 : {Counter(y)}')
# print(f'train 분포 : {Counter(y_train)}')
# print(f'test 분포 : {Counter(y_test)}')
'''
전체 분포 : Counter({np.int64(1): 357, np.int64(0): 212})
train 분포 : Counter({np.int64(1): 285, np.int64(0): 170})
test 분포 : Counter({np.int64(1): 72, np.int64(0): 42})
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
for clf in [logi, knn, tree]:
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    print(f'{clf.__class__.__name__} 정확도 : {accuracy_score(y_test, pred):.4f}')
'''
Pipeline 정확도 : 0.9912 logi
Pipeline 정확도 : 0.9737 knn
DecisionTreeClassifier 정확도 : 0.8772
'''

voting.fit(x_train, y_train)
vpred = voting.predict(x_test)
print(f'voting 분류기 정확도 : {accuracy_score(y_test, vpred):.4f}')
# voting 분류기 정확도 : 0.9649
# 위에 세 개 평균 값이 voting값

# 옵션 : 교차 검증으로 안정성 확인
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12) # kfold 사용해도 무방
cv_score = cross_val_score(voting, x, y, cv=cv, scoring='accuracy')
print(f'voting 5겹 cv 평균 : {cv_score.mean():.4f} ∓{cv_score.std():.4f}')

