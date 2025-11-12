'''
seaborn : matplotlib의 기능 보강 모듈
'''
import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')
# titanic.info()

sns.boxplot(y='age', data=titanic, palette='Paired')
plt.show()

# sns.displot(titanic['age']) # 막대
sns.kdeplot(titanic['age']) # 밀도
plt.show()

sns.relplot(x='who', y='age', data=titanic) # 범주형
plt.show()

sns.countplot(x='pclass', data=titanic, palette='viridis')
plt.show()

t_pivot = titanic.pivot_table(index='class', columns='sex',aggfunc='size')
print(t_pivot)
# sex     female  male
# class
# First       94   122
# Second      76   108
# Third      144   347

# 히트맵 : 색깔로 데이터 표현
sns.heatmap(t_pivot, cmap=sns.light_palette('gray', as_cmap=True), annot=True, fmt='d')
plt.show()