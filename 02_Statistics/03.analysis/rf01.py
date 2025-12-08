'''
## RandomForest 분류/예측 알고리즘
- 분류 알고리즘으로 titanic dataset 사용해서 이진분류
- 데이터 샘플링(bootstrap)을 통해 모델을 학습시키고, 결과를 집계(aggregation)하는 방법이 Bagging이다.
- 참고 : 우수한 성능을 원한다면 Boosting, 오버피팅이 걱정된다면 Bagging 추천
'''
# titanic dataset : feature (pclass, age, sex), label (surived)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv')
print(df.head())
df.info()
print(df.isnull().any())

df = df.dropna(subset=['Pclass','Age','Sex'])
print(df.shape)

# feature, label 분리
df_x = df[['Pclass','Age','Sex']].copy()
print(df_x.head()) # Sex를 더미화

# 더미화 클래스
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_x.loc[:,'Sex']=encoder.fit_transform(df_x['Sex'])
print(df_x.head())

df_y = df['Survived']
print(df_y.head(2))

train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3, random_state=12)

# 모델 생성
model = RandomForestClassifier(criterion='entropy', n_estimators=500)
model.fit(train_x, train_y)
pred = model.predict(test_x)
print(f'예측값 : {pred[:10]}')
print(f'실제값 : {test_y[:10]}')
print(f'맞춘 개수 : {sum(test_y == pred)}')
print(f'전체 대비 맞춘 비율 : {sum(test_y == pred)/len(test_y)}')
print(f'분류 정확도 : {accuracy_score(test_y,pred)}')

# k-fold
cross_vali = cross_val_score(model, df_x, df_y, cv =5 )
print(cross_vali)
print(f'교차 검증 평균 정확도 : {np.mean(cross_vali):.5f}')

# 중요변수 확인
print(f'특성(변수) 중요도 : {model.feature_importances_}')

# 시각화
import matplotlib.pyplot as plt
n_features = df_x.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.xlabel('feature_importance score')
plt.ylabel('features')
plt.yticks(np.arange(n_features), df_x.columns)
plt.ylim(-1, n_features)
plt.show()
plt.close()