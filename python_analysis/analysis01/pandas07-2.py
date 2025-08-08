import pandas as pd

url = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/human.csv'
df = pd.read_csv(url, encoding='utf-8')
# print(df.head())

'''
- Group이 NA인 행은 삭제
- Career, Score 칼럼을 추출하여 데이터프레임을 작성
- Career, Score 칼럼의 평균계산
'''
print(df.columns)
df.columns = df.columns.str.strip()
df['Group'] = df['Group'].str.strip()
df.replace('NA', pd.NA, inplace=True)
df.dropna(subset=['Group'], inplace=True)
print(df)

df = df[['Career', 'Score']].copy()
print(df)

print(df.mean())


'''
2) tips.csv 파일을 읽어 아래와 같이 처리하시오.
     - 파일 정보 확인
     - 앞에서 3개의 행만 출력
     - 요약 통계량 보기
     - 흡연자, 비흡연자 수를 계산  : value_counts()
     - 요일을 가진 칼럼의 유일한 값 출력  : unique()
          결과 : ['Sun' 'Sat' 'Thur' 'Fri']
'''

url2 = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tips.csv'
df2 = pd.read_csv(url2, encoding='utf-8')

print(df2.info())
print(df2.head(3))
print(df2.describe())

print(df2['smoker'].value_counts())

unique_cols = df2['day'].unique()
print(unique_cols)