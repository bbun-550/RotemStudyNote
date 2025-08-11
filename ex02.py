'''
from pandas import DataFrame
data = {"a": [80, 90, 70, 30], "b": [90, 70, 60, 40], "c": [90, 60, 80, 70]}

칼럼(열)의 이름을 순서대로 "국어", "영어", "수학"으로 변경한다.
아래 문제는 제시한 columns와 index 명을 사용한다.
1) 모든 학생의 수학 점수를 출력하기
2) 모든 학생의 수학 점수의 표준편차를 출력하기
3) 모든 학생의 국어와 영어 점수를 Series 타입이 아니라 DataFrame type으로 출력하기 (배점:10)
'''
from pandas import DataFrame
import numpy as np

data = {'a': [80, 90, 70, 30], 'b': [90, 70, 60, 40], 'c': [90, 60, 80, 70]}
print(data, type(data))

df = DataFrame(data)
print(DataFrame(data))

# print(df['math'].std())

# print(df.groupby['kor']('eng'))




# df = DataFrame(data, columns=['국어','영어','수학'])
# print(df['수학'])

# print(df['수학'].std())

print(df[['a','b']])
