'''
## 의사결정나무 Decision Tree - CART(Classification And Regression Tree)
- 예측 분류가 모두 가능하나 분류가 주 목적이다.
- 비모수 검정을 한다. 
    - 선형성, 정규성, 등분산성 가정이 필요없다.
- 단점이 있다면, 유의수준 판단 기준이 없다. 과적합으로 예측 정확도가 낮을 수 있다.
'''
import matplotlib.pyplot as plt
from sklearn import tree

# 키와 머리카락 길이로 성별 구분 모델 작성
x = [[180,15],[177,42],[156,35],[174,65],[161,28],[160,5],[170,12],[176,75],[170,22],[175,28]]
y = ['man','woman','woman','man','woman','woman','man','man','man','woman'] # 범주형 데이터
feature_names = ['height', 'hair length']
class_names = ['man','woman']


# 모델
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
# citerion = 'gini'/'entropy'
# max_depth 데이터가 많아지면 depth가 커진다(불순도 높은 데이터) : 트리의 최대 깊이를 지정
# min_samples_split
# min_samples_leaf
# min_weight_fractin_leaf
# max_features 

model.fit(x,y)

print(f'훈련 데이터 정확도 : {model.score(x,y):.3f}')
print(f'예측 결과 : {model.predict(x)}')
print(f'실제 결과 : {y}')

# 새로운 자료로 분류 예측
new_data = [[199,60]]
print(f'예측 결과 : {model.predict(new_data)}')

# 시각화
plt.figure(figsize=(10,6))
tree.plot_tree(model, feature_names=feature_names, class_names=class_names,
               filled=True, rounded=True, fontsize=10)
plt.show()