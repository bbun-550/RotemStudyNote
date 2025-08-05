from pandas import Series, DataFrame
import numpy as np

s1 = Series([1,2,3], index = ['a','b','c'])
s2 = Series([1,2,3,4], index = ['a','b','c','d'])
# print(s1)
# print(s2)

# Series 연산
print(s1 + s2)
print(s1.add(s2))
print(s1.multiply(s2))
print(s1.divide(s2))

print('-'*40)

df1 = DataFrame(np.arange(9).reshape(3,3), columns=list('kbs'), index=['서울','대구','대전'])
df2 = DataFrame(np.arange(12).reshape(4,3), columns=list('kbs'), index=['서울','대전','제주','수원'])
print(df1)
print(df2)
#     k  b  s
# 서울  0  1  2
# 대구  3  4  5
# 대전  6  7  8
#     k   b   s
# 서울  0   1   2
# 대전  3   4   5
# 제주  6   7   8
# 수원  9  10  11

# DataFrame 연산
print(df1+df2)
print(df1.add(df2, fill_value=1)) # NaN은 0으로 채운 후 연산에 참여
print(df1.multiply(df2, fill_value=0))
print(df1.divide(df2, fill_value=0))

print('-'*40)

ser1 = df1.iloc[0] # 빠져나와서 series가 됨
print(ser1) 
# k    0
# b    1
# s    2
# Name: 서울, dtype: int64

print(df1 - ser1) # broadcasting 연산
#     k  b  s
# 서울  0  0  0
# 대구  3  3  3
# 대전  6  6  6
print('-'*40)

'''
결측치, 기술적 통계 관련 함수
'''
# 결측치
df = DataFrame([[1.4, np.nan], [7,-4.5], [np.nan, None], [0.5, -1]], columns=['one','two']) # 4행 2열
print(df)
#    one  two
# 0  1.4  NaN
# 1  7.0 -4.5
# 2  NaN  NaN
# 3  0.5 -1.0

print(df.isnull())
print(df.notnull())

print(df.drop(0)) # 특정 행 삭제(NaN과 관계없음. 그냥 지워버려)

print(df.dropna()) # na값이 포함된 모든 행 삭제
print(df.dropna(how='any')) # default 값임. 어느 행이든 지우는
#    one  two
# 1  7.0 -4.5
# 3  0.5 -1.0

print(df.dropna(how='all'))
#    one  two
# 0  1.4  NaN
# 1  7.0 -4.5
# 3  0.5 -1.0

print(df.dropna(subset=['one'])) # one이라는 열에 NaN이 있으면 날려버린다
#    one  two
# 0  1.4  NaN
# 1  7.0 -4.5
# 3  0.5 -1.0

print(df.dropna(axis='rows'))
print(df.dropna(axis='columns'))

print(df.fillna(0)) # 지우기 아까울 때, NaN 값을 0으로 채운다. 평균, 대푯값, 최빈값, 이전값, 다음값 ... 채울 수 있음
#    one  two
# 0  1.4  0.0
# 1  7.0 -4.5
# 2  0.0  0.0
# 3  0.5 -1.0
print('-'*40)

# 기술 통계 관련 method
print(df.sum()) # NaN은 연산에 찾여하지 않음. 열의 합 출력
print(df.sum(axis=0))
# one    8.9
# two   -5.5
print(df.sum(axis=1)) # 행단위 합

print(df.describe()) # 요약 통계량 보여준다
#             one       two
# count  3.000000  2.000000
# mean   2.966667 -2.750000
# std    3.521837  2.474874
# min    0.500000 -4.500000
# 25%    0.950000 -3.625000
# 50%    1.400000 -2.750000
# 75%    4.200000 -1.875000
# max    7.000000 -1.000000

print(df.info()) # 구조가 궁금해? 구조확인
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 4 entries, 0 to 3
# Data columns (total 2 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   one     3 non-null      float64
#  1   two     2 non-null      float64
# dtypes: float64(2)
# memory usage: 196.0 bytes

'''
재구조화
구간 설정
그룹별 연산
agg 함수
'''
df = DataFrame(1000 + np.arange(6).reshape(2,3), index=['서울','대전'], columns=['2020','2021','2022'])
print(df)
#     2020  2021  2022
# 서울  1000  1001  1002
# 대전  1003  1004  1005

print(df.T) # Transpose. 서울, 대전이 열로 바뀜. 재구조화
#         서울    대전
# 2020  1000  1003
# 2021  1001  1004
# 2022  1002  1005

# stack, unstack
df_row = df.stack() # 열 -> 행으로 변경. 열 쌓기
print(df_row)
# 서울  2020    1000
#     2021    1001
#     2022    1002
# 대전  2020    1003
#     2021    1004
#     2022    1005

df_col = df_row.unstack() # 행 -> 열로 변경
print(df_col)
print('-'*40)

# 구간 설정
# 연속형 자료를 범주화시키는 것
import pandas as pd
price = [10.3, 5.5, 7.8, 3.6]
cut = [3,7,9,11] # 구간 기준값
result_cut = pd.cut(price, cut)
print(result_cut)
# [(9, 11], (3, 7], (7, 9], (3, 7]]
# Categories (3, interval[int64, right]): [(3, 7] < (7, 9] < (9, 11]]
# \( 초과, \] 이하

print(pd.value_counts(result_cut))
# (3, 7]     2 # 3초과 7이하
# (7, 9]     1
# (9, 11]    1

datas = pd.Series(np.arange(1,1001))
print(datas.head(3))
print(datas.tail(3))

# 범주화
result_cut2 = pd.qcut(datas, 3) # 3개 구간으로 나눈다
print(result_cut2)
# Categories (3, interval[float64, right]): [(0.999, 334.0] < (334.0, 667.0] < (667.0, 1000.0]]

print(pd.value_counts(result_cut2))
# (667.0, 1000.0]    334
# (0.999, 334.0]     333
# (334.0, 667.0]     333
print('-'*40)

group_col = datas.groupby(result_cut2)
# print(group_col) # 객체야 오케이
print(group_col.agg(['count','mean','std','min'])) # 그룹별 소계 구할 수 있음
#                  count   mean        std  min
# (0.999, 334.0]     333  167.0  96.273049    1
# (334.0, 667.0]     333  500.0  96.273049  334
# (667.0, 1000.0]    334  833.5  96.561725  667

# agg 함수 만들기
def myFunc(gr):
    return {'count': gr.count(),
            'mean': gr.mean(),
            'std': gr.std(),
            'min': gr.min()            
            }

print(group_col.apply(myFunc)) # 함수를 실행하는 함수 .apply
print(group_col.apply(myFunc).unstack()) # .agg랑 똑같이 나옴
#                  count   mean        std    min
# (0.999, 334.0]   333.0  167.0  96.273049    1.0
# (334.0, 667.0]   333.0  500.0  96.273049  334.0
# (667.0, 1000.0]  334.0  833.5  96.561725  667.0

# 클래스?





