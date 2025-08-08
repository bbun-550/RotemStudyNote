'''
reindexing 재색인
'''
from pandas import Series, DataFrame
import numpy as np

# Series의 재색인
data = Series([1,3,2], index=[1,4,2])
print(data)
data2 = data.reindex((1,2,4))
print(data2)

# 재색인할 때, 값 채우기
data3 = data2.reindex([0,1,2,3,4,5])
print(data3) 
### 대응값이 없는(NaN) 인덱스는 결측값인데 777로 채워기
data3 = data2.reindex([0,1,2,3,4,5], fill_value=777)
print(data3) 
print('-'*40)

# 기존 데이터로 결측치 채우기
data3 = data2.reindex([0,1,2,3,4,5], method='ffill') # front fill 앞에 값으로 뒤에 값 채운다
print(data3) 
data3 = data2.reindex([0,1,2,3,4,5], method='pad') # 
print(data3) 
print('-'*40)

data3 = data2.reindex([0,1,2,3,4,5], method='bfill') # back fill 뒤에 값으로 앞에 값 채운다
print(data3) 
data3 = data2.reindex([0,1,2,3,4,5], method='backfill') # 
print(data3) 
print('-'*40)

# bool 처리, 슬라이싱 관련 method : loc(), iloc()
df = DataFrame(np.arange(12).reshape(4,3), index=['1월', '2월','3월','4월'],
        columns=['강남','강북','서초'])
print(df)
print('-'*40)
print(df['강남'] > 3) # True, False 출력
# 1월    False
# 2월    False
# 3월     True
# 4월     True
# Name: 강남, dtype: bool

print('-'*40)
print(df[df['강남'] > 3]) # 조건 부합하는 값 출력
#     강남  강북  서초
# 3월   6   7   8
# 4월   9  10  11
print('-'*40)

'''
복수 인덱싱
loc() : 라벨 지원(문자로 인덱싱할 수 있다), 숫자 안씀
iloc() : 숫자 지원
'''
print(df.loc[:'2월']) # 2월 이하 나와
#     강남  강북  서초
# 1월   0   1   2
# 2월   3   4   5

print(df.loc[:'2월', ['서초']]) # 2월 이하 서초만 보고싶어
#     서초
# 1월   2
# 2월   5

print(df.iloc[2]) # 2행을 출력해라
print(df.iloc[2, :]) # 2행의 모든 열을 출력해라
# 강남    6
# 강북    7
# 서초    8

print(df.iloc[:3]) # 3행 미만 출력해라
#     강남  강북  서초
# 1월   0   1   2
# 2월   3   4   5
# 3월   6   7   8

print(df.iloc[:3, 2], type(df.iloc[:3, 2])) # 3행 미만 2열만 출력해라
# 1월    2
# 2월    5
# 3월    8
# Name: 서초, dtype: int64 <class 'pandas.core.series.Series'>