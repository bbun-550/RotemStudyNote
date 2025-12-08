# 세계 정치인들 얼굴 사진을 이용한 분류 모델
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import numpy as np

faces = fetch_lfw_people(min_faces_per_person=60, color=False, resize=0.5) # min_faces_per_person : 인물 당 사진 수
# print(faces.DESCR) # 셜명서
'''
**Data Set Characteristics:**
=================   =======================
Classes                                5749
Samples total                         13233
Dimensionality                         5828
Features            real, between 0 and 255
=================   =======================
'''

# print(faces.data)
# print(f'faces.data shape : {faces.data.shape}') # (1348, 2914)
# print(faces.target)
# print(faces.target_names) # ['Ariel Sharon' 'Colin Powell' 'Donald Rumsfeld' 'George W Bush' 'Gerhard Schroeder' 'Hugo Chavez' 'Junichiro Koizumi' 'Tony Blair']
# print(f'faces.images shape : {faces.images.shape}') # (1348, 62, 47) 가로 67 , 세로 47 픽셀로 된 그림이 1348장이 있다.

# 얼굴 이미지 확인
# print(faces.images[1345])
# print(faces.target_names[faces.target[1345]])
# 그림 그리기
# plt.imshow(faces.images[1345], cmap='bone') # bone : 흑백
# plt.show()
# plt.close()

'''
# 이미지 여러장 표시
fig, ax = plt.subplots(3,5)
# print(fig) # Figure(640x480)
# print(len(ax.flat)) # 15
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]]) # 눈금 제거, 축에 이름 표시
plt.tight_layout()
plt.show()
plt.close()
'''
# 주성분분석으로 이미지 차원축소시켜 분류 작업 진행 - 
m_pca = PCA(n_components=150, whiten=True, random_state=0)
x_low = m_pca.fit_transform(faces.data)
print(f'x_low : {x_low}\n{x_low.shape}') # (1348, 150) 2914 -> 150개로 줄였다. 이미지에 영향을 많이 줄 것은 데이터만 남겼다.

m_svc = SVC(C=1.0)
model = make_pipeline(m_pca, m_svc) # 하나로 묶어서 순차적으로 진행한다. 안에서 fit_transform도 해준다.

# train/test
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(faces.data,faces.target, test_size=0.2, random_state=1)

model.fit(train_x, train_y)
pred = model.predict(test_x)
print(f'예측값 : {pred[:10]}')
print(f'실제값 : {test_y[:10]}')
print()

from sklearn.metrics import classification_report # 분류모델 성능 요약해주는 함수이다.
print(classification_report(test_y, pred, target_names=faces.target_names))

# 오차행렬
from sklearn.metrics import confusion_matrix, accuracy_score
matrix = confusion_matrix(test_y, pred)
print(f'confusion matrix :\n{matrix}')
print(f'acc : {accuracy_score(test_y, pred):.4f}') # acc : 0.8037

# 분류 결과 시각화
# - test_x[0] 하나만 확인한다.
print(test_x[0], test_x.shape) # (270, 2914)
print(test_x[0].reshape(62,47)) # 픽셀 가로62, 세로47 모양으로 잡아준다.
# 이미지 출력시 이처럼 1차원을 2차원으로 변환해준다.
plt.subplots(1,1)
plt.imshow(test_x[0].reshape(62,47), cmap='bone')
plt.title(f'{faces.target_names[pred[0]]}')
plt.show()
plt.close()

fig, ax = plt.subplots(4,6)
for i, axi in enumerate(ax.flat):
    axi.imshow(test_x[i].reshape(62,47), cmap='bone')
    axi.set(xticks=[], yticks=[]) # 눈금 제거, 축에 이름 표시
    axi.set_xlabel(faces.target_names[pred[i]].split()[-1],
                  color='black' if pred[i] == test_y[i] else 'red') # 맞추면 이름 검정색으로 출력, 틀리면 빨간색으로 출력된다.
fig.suptitle('pred result', size=14)
plt.tight_layout()
plt.show()
plt.close()

# 오차 행렬 시각화
import seaborn as sns
sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.xlabel('true(real) label')
plt.ylabel('pred label')
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
plt.close()