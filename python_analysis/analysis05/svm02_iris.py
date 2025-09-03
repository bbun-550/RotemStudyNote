from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # 표준화
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic') # macOS : applegothic / windows : malgun gothic
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.colors import ListedColormap # 색깔 출력
import pickle # 모델 저장

from sklearn.linear_model import LogisticRegression # 다중 클래스(python class 아님, 종속변수 label y class) 지원

iris = datasets.load_iris()
# print(iris['data'])
print(np.corrcoef(iris.data[:,2], iris.data[:,3])[0][1]) # 상관관계 : 0.96286

x = iris.data[:,[2,3]] # petal.length / petal.width 작업 : type matrix 2차원
y = iris.target # type vector 1차원
# print(x[:3])
# print(y[:3], set(y))

# train / test split 7:3
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (105, 2) (45, 2) (105,) (45,)
'''
## Scaling (데이터 표준화 standardscaler - 최적화 과정에서 안정성, 수렴 숙도 향상, 오버플로우/언더플로우 방지 효과)
#- 데이터값이 클 경우, 데이터 표준화했을 때 모델이 잘 나온다. 원본으로 해보고 잘 안나올 경우에 표준화한다.
print(x_train[:3])
sc = StandardScaler()

sc.fit(x_train); sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test) # 독립변수만 스케일링
print(x_train[:3])

# 스케일링 원상복귀
inver_x_train = sc.inverse_transform(x_train)
print(inver_x_train[:3])
'''
# 분류 모델 생성
# - C 속성 : L2 규제 - 모델에 패널티 적용(tunning parameter). 숫자 값을 조정해가며 분류 정확도를 확인한다(최적의 L2를 찾는다. 10의 배수). 기본값은 1.0이다
# 값이 작을수록 더 강한 정규화 규제를 한다.
# model = LogisticRegression(C=0.1, random_state=0, verbose=0) # solver='lbfgs' 기본값 / verbose 학습진행 상황 보고
from sklearn import svm
model = svm.SVC(C=0.1)
model.fit(x_train,y_train) # 지도학습 supervised learning : 문제지와 답지를 제공

# 분류 예측 - 모델 성능 파악하기 위해
y_pred = model.predict(x_test)
print(f'예측값 : {y_pred}')
print(f'실제값 : {y_test}')
print('총 개수:%d, 오류수:%d'%(len(y_test), (y_test != y_pred).sum()))

# 분류 정확도 확인 3가지
# 1. accuracy_score
print('분류 정확도 1 : %.3f'%accuracy_score(y_test, y_pred))

# 2. crosstable
con_mat = pd.crosstab(y_test, y_pred, rownames=['예측값'], colnames=['관측값'])
print(f'분류 정확도 2 : {(con_mat[0][0] + con_mat[1][1] + con_mat[2][2])/len(x_test):.3f}')

# 3. 직접 비교
print(f'test : {model.score(x_test, y_test)}')
print(f'train : {model.score(x_train, y_train)}') # 두 개의 값 차이가 크면 과적합 의심
'''
# 결과
분류 정확도 1 : 0.978
분류 정확도 2 : 0.978
test : 0.9777777777777777
train : 0.9428571428571428
'''


# 시각화
def plot_decisionFunc(X, y, classifier, test_idx=None, resulution=0.02, title=''):
    # test_idx : test 샘플의 인덱스
    # resulution : 등고선 오차 간격
    markers = ('s','x','o','^','v')   # 마커(점) 모양 5개 정의함
    colors = ('r', 'b', 'lightgray', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])  # 색상팔레트를 이용
    # print(cmap.colors[0], cmap.colors[1])
    
    # surface(결정 경계) 만들기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 좌표 범위 지정
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # 격자 좌표 생성
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resulution), \
                         np.arange(x2_min, x2_max, resulution))
    
    # xx, yy를 1차원배열로 만든 후 전치한다. 이어 분류기로 클래스 예측값 Z얻기
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)  # 원래 배열(격자 모양)로 복원

    # 배경을 클래스별 색으로 채운 등고선 그리기
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx), \
                    marker=markers[idx], label=cl)
    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(x=X[:, 0], y=X[:, 1], color=[], \
                    marker='o', linewidths=1, s=80, label='test')
    plt.xlabel('꽃잎길이')
    plt.ylabel('꽃잎너비')
    plt.legend()
    plt.title(title)
    plt.show()

# train과 test 모두를 한 화면에 보여주기 위한 작업 진행
# train과 test 자료 수직 결합(위 아래로 이어 붙임 - 큰행렬 X 작성)
x_combined_std = np.vstack((x_train, x_test))   # feature
# 좌우로 이어 붙여 하나의 큰 레이블 벡터 y 만들기
y_combined = np.hstack((y_train, y_test))    # label
plot_decisionFunc(X=x_combined_std, y=y_combined, classifier=model, \
                  test_idx = range(100, 150), title='scikit-learn 제공')

