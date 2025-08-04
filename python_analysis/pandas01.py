''' 
pandas : 행과 열이 단순 정수형 index가 아닌 label로 식별되는 numpy의 구조화된 배열
        보완(강)한 모듈이다. 고수준의 자료구조(시계열 축약연산, 누락데이터처리, SQL, 시각화...)
'''
import pandas as pd
from pandas import Series
from pandas import DataFrame
import numpy as np

# Series : 일련의 객체를 담을 수 있는 1차원 배열과 유사한 자료구조로 색인을 갖음
# List, array로 부터 만들 수 있다
obj = pd.Series([3,7,-5,4]) # set{}은 순서가 없기 때문에 오류. list[],tupe() 순서 있음 그래서 사용 가능
print(obj, type(obj)) # dtype: int64 <class 'pandas.core.series.Series'>
# 자동 인덱스(명시적) <> numpy(묵시적)
print(obj[1])

obj2 = pd.Series([3,7,-5,4], index=['a', 'b', 'c', 'd'])
print(obj2)
print(obj2['c'])
print(obj2.sum(), np.sum(obj2)) # python 제공 sum, numpy 제공 sum(더 빠름)
print(obj2.values) # [ 3  7 -5  4]
print(obj2.index) # Index(['a', 'b', 'c', 'd'], dtype='object')

# 슬라이싱
print(obj2['a'])
# print(obj2[0]) # 경고 출력 'use `ser.iloc[pos]`
print(obj2[['a']]) # a 3 인덱스와 값이 같이 나옴
print(obj2[['a','c']])
print(obj2['a':'c'])
print(obj2[3])
print(obj2.iloc[3])
print(obj2[[2,3]])
print(obj2.iloc[[2,3]])
print(obj2 > 0)
print('a' in obj2)

# dict type으로 Series 객체 생성
# dict 순서 있음
names = {'mouse' :5000, 'keyboard' :25000, 'monitor' :450000}
print(names, type(names)) # <class 'dict'>
obj3 = Series(names) # key가 인덱스가 됨
print(obj3, type(obj3)) # <class 'pandas.core.series.Series'>
obj3.index = ['마우스', '키보드', '모니터']
print(obj3)
print(obj3['마우스'])
print(obj3.iloc[0])

obj3.name = '상품가격' # Name: 상품가격
print(obj3)
print('=' * 50)

# Dataframe : Series 객체가 모여서 표를 구성
df = DataFrame(obj3)
print(df)

data = { # dict type
    'irum' :['홍길동', '한국인', '신기해', '공기밥', '한기범'],
    'juso' :('역삼동','신당동', '신사동', '양평동','삼청동'),
    'nai' :[23,25,33,30,35]
}
frame = DataFrame(data)
print(frame)

print(frame['irum']) # 이 방식을 더 좋아함
print(frame.irum)
print(type(frame.irum)) # <class 'pandas.core.series.Series'> : 열 단위로 추출하면 series
print(DataFrame(data, columns=['juso', 'irum','nai']))

# data에 NaN(Not a Number)을 넣기
frame2 = DataFrame(data, columns=[ 'irum','nai', 'juso', 'tel'],
                   index=['a','b','c','d','e'])
print(frame2)
frame2['tel'] = '111-1111'
print(frame2)

val = Series(['222-2222', '333-3333','444-4444'], index=['b','c','e'])
frame2.tel = val
print(frame2)

print(frame2.T)

print(frame2.values, type(frame2.values)) # 2차원 배열 형식. 중복 리스트 반환. <class 'numpy.ndarray'> 문자열 취급
print(frame2.values[0,1]) # 0행 1열
print(frame2.values[0:2]) # 0행 -1행

# 행 / 열 삭제
frame3 = frame2.drop('d', axis=0) # 인덱스가 d 행 삭제. axis 안쓰면 0 default
print(frame3)

frame4 = frame2.drop('tel', axis=1) # tel 열 삭제
print(frame4)

# 정렬
print(frame2.sort_index(axis=0, ascending=False)) # descending sort. 행 단위 내림차순 정렬
print(frame2.sort_index(axis=1, ascending=True)) # ascending True 생략 가능. 열 단위 오름차순 정렬

print(frame2['juso'].value_counts()) # juso 개수

#문자열 자르기
data ={
    'juso' :['강남구 역삼동', '중구 신당동', '강남구 대치동'],
    'inwon' :[23, 30, 15]
}
fr = pd.DataFrame(data)
print(fr)
result1 = Series([x.split()[0] for x in fr.juso]) # split 공백 구분자로 문자열 분리
result2 = Series([x.split()[1] for x in fr.juso])
print(result1, result1.value_counts())
print('-'*20)
print(result2, result2.value_counts())

