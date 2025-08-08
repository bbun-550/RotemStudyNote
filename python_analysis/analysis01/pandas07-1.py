import pandas as pd


url = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv'
df = pd.read_csv(url, sep=',')
# print(df.head(2))

bins = [1, 20, 35, 60, 150]
labels = ["소년", "청년", "장년", "노년"]
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
print(df[['PassengerId','Name','Sex','Age','Age_Group']].head(3))

# df.to_csv('titanic_data.csv', index=False, encoding='utf-8')

print('출력 결과 샘플1 :')
print(pd.pivot_table(df,values='Survived', index='Sex', columns='Pclass', aggfunc='mean'))
# Pclass         1         2         3
# Sex
# female  0.968085  0.921053  0.500000
# male    0.368852  0.157407  0.135447


print('출력 결과 샘플2 :')
pd.options.display.float_format = '{:.2f}'.format
print(pd.pivot_table(df, values='Survived', index=['Sex','Age_Group'], columns='Pclass', aggfunc='mean',observed=False))
# Pclass              1    2    3
# Sex    Age_Group
# female 소년        0.93 1.00 0.47
#        청년        0.97 0.92 0.50
#        장년        0.97 0.85 0.14
#        노년        1.00  NaN 1.00
# male   소년        0.50 0.36 0.18
#        청년        0.54 0.06 0.16
#        장년        0.37 0.04 0.07
#        노년        0.08 0.33 0.00