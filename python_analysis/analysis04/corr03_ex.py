'''
상관관계 문제)
https://github.com/pykwon/python - Advertising.csv
tv,radio,newspaper 간의 상관관계를 파악하시오. 
그리고 이들의 관계를 heatmap 그래프로 표현하시오. 
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

raw = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv')
# print(raw.head())

table = raw[['no','tv','radio','newspaper']]
table = table.set_index('no')
# print(table.head())

print(table.corr())
#                  tv     radio  newspaper
# tv         1.000000  0.054809   0.056648
# radio      0.054809  1.000000   0.354104
# newspaper  0.056648  0.354104   1.000000

sns.heatmap(table.corr(), cmap='Greys', annot=True)
plt.title('Advertising.csv')
plt.show()
plt.close()