'''
iris dataset
꽃받침과 꽃잎의 너비와 길이로 꽃의 종류를 3가지로 구분해 놓은 데이터
각 그룹 당 50개, 총 150개 데이터
'''
import pandas as pd
import matplotlib.pyplot as plt

iris_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/iris.csv')
# print(iris_data.head())
# iris_data.info()
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   Sepal.Length  150 non-null    float64
#  1   Sepal.Width   150 non-null    float64
#  2   Petal.Length  150 non-null    float64
#  3   Petal.Width   150 non-null    float64
#  4   Species       150 non-null    object
# dtypes: float64(4), object(1)

# 산포도, 산점도 그려보자
# 꽃받침 길이, 꽃잎 길이
plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'])
plt.xlabel('Sepal.Length')
plt.ylabel('Petal.Length')
plt.show()

# 색깔로 종류별 구분해보자
print(iris_data['Species'].unique()) # 종류 출력
print(set(iris_data['Species']))

cols = []
for s in iris_data['Species']:
    choice = 0
    if s == 'setosa': choice = 1
    elif s == 'versicolor':choice = 2
    elif s == 'virginica' : choice = 3
    cols.append(choice)

plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'], c=cols)
plt.xlabel('Sepal.Length')
plt.ylabel('Petal.Length')
plt.title('types of flowers')
plt.show()


# 분포도, 산점도 그래프 같이 그려보자
iris_col = iris_data.loc[:,'Sepal.Length':'Petal.Length']
from pandas.plotting import scatter_matrix # pandas의 시각화 기능 활용

scatter_matrix(iris_col, diagonal='kde') # 후자 밀도추정 곡선
plt.show()

# seaborn 사용
import seaborn as sns
sns.pairplot(iris_data, hue='Species', height=2) # hue : category 준다
plt.show()

# rug plot
x = iris_data['Sepal.Length'].values
sns.rugplot(x)
plt.show()

# kernel density 부드러운 곡선
sns.kdeplot(x)
plt.show()









