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
print('-'*40)
print(df[df['강남'] > 3]) # 조건 부합하는 값 출력




