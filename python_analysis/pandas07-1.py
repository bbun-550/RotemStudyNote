import pandas as pd


url = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv'
df = pd.read_csv(url, sep=',')
# print(df.head(2))

bins = [1, 20, 35, 60, 150]
labels = ["소년", "청년", "장년", "노년"]
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
# print(df[['PassengerId','Name','Sex','Age','Age_Group']].head(3))

# df.to_csv('titanic_data.csv', index=False, encoding='utf-8')

print('출력 결과 샘플1 :')
print(pd.pivot_table(df,values='Survived', index='Sex', columns='Pclass', aggfunc='mean'))

print('출력 결과 샘플2 :')
pd.options.display.float_format = '{:.2f}'.format
print(pd.pivot_table(df, values='Survived', index=['Sex','Age_Group'], columns='Pclass', aggfunc='mean'))
