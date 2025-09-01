'''
## 과적합 방지 처리 방법
- train/test split
- KFold
- GridSearchCV
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score

iris = load_iris()
print(iris.keys()) # ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']
train_data = iris.data
train_label = iris.target
print(train_data[:3]) # 
print(train_label[:3]) # 앞 50개는 setosa

# 일반 분류 모델
dt_clf = DecisionTreeClassifier()
dt_clf.fit(train_data, train_label) # 학습
pred = dt_clf.predict(train_data)
print(f'예측값 : {pred}')
print(f'실제값 : {train_label}')
print(f'분류정확도 : {accuracy_score(train_label,pred)}')
# 분류정확도 : 1.0 => overfitting 좋지 않는 숫자. 포용성이 떨어진다. 일반화된 모델이 아니다.(특정 데이터에 최적화되어 있는 모델이다.)

# 과적합 발생
print('과적합 방지 방법 1 : train/test로 분리')
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, shuffle=True, random_state=121)

# print(x_train.shape, y_train.shape) # (105, 4) (105,)
dt_clf.fit(x_train, y_train) # train으로 학습
pred2 = dt_clf.predict(x_test) # test로 예측(성능하기 위함)
print(f'예측값 : {pred2}')
print(f'실제값 : {y_test}')
print(f'분류정확도2 : {accuracy_score(y_test,pred2)}')
# 분류정확도2 : 0.9555 => 과적합 해소, 일반화된 모델, 포용성 있는 모델이 작성되었다.

print('과적합 방지 방법 2 : 교차검정 cross validation')
# KFold 교차 검증이 가장 일반적이다.
# train dataset에 대해 k 개의 data fold set을 만들어서 k 번 만큼 학습 도중에 검증 평가를 수행하는 방법이다.
# train/test 잘랐으면 train에 대해서 k번 잘라서 검증 평가를 수행한다.
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(criterion='entropy',random_state=123)
kfold = KFold(n_splits=5) # kfold 랑 train/test 별개이다. 같이 써도 되고 따로 써도 된다.
cv_acc = []

# print(f'iris shape : {features.shape}') # (150, 4)
n_iter = 0
for train_idx, test_idx in kfold.split(features):
    # print(f'n_iter : {n_iter}')
    # print(f'train_idx : {len(train_idx)}')
    # print(f'test_idx : {len(test_idx)}')
    # n_iter += 1
    
    # kfold.split으로 변환된 인덱스를 이용해서 학습용, 검증용 데이터 추출
    xtrain, xtest = features[train_idx], features[test_idx]
    ytrain, ytest = label[train_idx], label[test_idx]

    # 학습 및 예측
    dt_clf.fit(xtrain, ytrain) # train
    pred = dt_clf.predict(xtest) # test (사실은 validation)
    n_iter += 1

    # 반복할 때마다 정확도 측정
    acc = np.round(accuracy_score(ytest, pred),3)
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]
    print(f'반복수 : {n_iter}, 교차검증 정확도 : {acc}, 학습데이터 수: {train_size}, 검증데이터 수: {test_size}')
    print(f'반복수: {n_iter}, 검증인덱스: {test_idx}')
    cv_acc.append(acc)

print(f'평균 검증 정확도 : {np.mean(cv_acc)*100:.2f}%')
# 평균 검증 정확도 : 92.00%

# Stratified Fold : 불균형한 분포를 가진 데이터 집합을 위한 k-fold 방식이다.
# - 대출사기, 스팸메일, 강우량, 코로나 백신 검사, 
from sklearn.model_selection import StratifiedKFold
features = iris.data
labels = iris.target

skfold = StratifiedKFold(n_splits=5) # kfold 랑 train/test 별개이다. 같이 써도 되고 따로 써도 된다.
cv_acc = []

# print(f'iris shape : {features.shape}') # (150, 4)
n_iter = 0
for train_idx, test_idx in skfold.split(features, labels):
    # print(f'n_iter : {n_iter}')
    # print(f'train_idx : {len(train_idx)}')
    # print(f'test_idx : {len(test_idx)}')
    # n_iter += 1
    
    # kfold.split으로 변환된 인덱스를 이용해서 학습용, 검증용 데이터 추출
    xtrain, xtest = features[train_idx], features[test_idx]
    ytrain, ytest = labels[train_idx], labels[test_idx]

    # 학습 및 예측
    dt_clf.fit(xtrain, ytrain) # train
    pred = dt_clf.predict(xtest) # test (사실은 validation)
    n_iter += 1

    # 반복할 때마다 정확도 측정
    acc = np.round(accuracy_score(ytest, pred),3)
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]
    print(f'반복수 : {n_iter}, 교차검증 정확도 : {acc}, 학습데이터 수: {train_size}, 검증데이터 수: {test_size}')
    print(f'반복수: {n_iter}, 검증인덱스: {test_idx}')
    cv_acc.append(acc)

print(f'평균 검증 정확도 : {np.mean(cv_acc)*100:.2f}%')


print('교차 검증 함수로 처리 ---')
data = iris.data
label = iris.target
score = cross_val_score(dt_clf, data, label, scoring='accuracy', cv=5)

print(f'교차 검증별 정확도 : {np.round(score, 2)}')
print(f'평균 검증 정확도 : {np.round(np.mean(score),2)}')

print('과적합 방지 방법 3 : GridSearchCV - 최적의 파라미터를 제공한다.')
parameters = {'max_depth':[1,2,3],'min_samples_split':[2,3]} # dict type
grid_dtree = GridSearchCV(dt_clf, param_grid=parameters, cv=3, refit=True) # refit=True(default) 재학습
grid_dtree.fit(x_train, y_train) # 자동으로 복수의 내부 모형을 생성 실행해 가며 최적의 파라미터를 찾는다.

import pandas as pd
scoreDF = pd.DataFrame(grid_dtree.cv_results_)
pd.set_option('display.max_columns', None)
print(scoreDF)
print(f'Best parameter : {grid_dtree.best_params_}') # Best parameter : {'max_depth': 2, 'min_samples_split': 2}
print(f'Best accuarcy : {grid_dtree.best_score_}') # Best accuarcy : 0.9583333333333334

# 최적의 parameter를 탑재한 모델이 제공
estimator = grid_dtree.best_estimator_ # decision tree classifier
pred = estimator.predict(x_test)
print(f'예측값 : {pred}')
print(f'테스트 데이터 정확도 : {accuracy_score(y_test,pred)}')


