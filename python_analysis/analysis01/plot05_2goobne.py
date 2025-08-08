import plot05_1goobne as goobne
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='malgun gothic') # 한글 깨짐 방지

df = goobne.df

# sns.boxplot(y='가격', data=df, palette='viridis')
# plt.show()

# sns.scatterplot(x='종류', y='가격', data=df,palette='viridis')
# plt.xlabel('종류')
# plt.ylabel('가격')
# plt.ylim([0,30000])
# plt.title('굽네 치킨')
# plt.show()

# sns.pairplot(df, hue='종류', height=3, palette='viridis')
# plt.show()

sns.boxplot(x='종류', y='가격', data=df, palette='pastel')
plt.title('굽네 메뉴 종류별 가격 분포 (Boxplot)')
plt.ylim([0, 30000])
plt.show()