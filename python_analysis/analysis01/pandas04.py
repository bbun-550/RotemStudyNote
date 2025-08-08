'''
데이터프레임 병합 Merge
'''
import pandas as pd
import numpy as np

df1 = pd.DataFrame({'data1':range(7), 'key':['b','b','b','c','a','a','b']})
#    data1 key
# 0      0   b
# 1      1   b
# 2      2   b
# 3      3   c
# 4      4   a
# 5      5   a
# 6      6   b

df2 = pd.DataFrame({'key':['a','b','d'], 'data2':range(3)})
#   key  data2
# 0   a      0
# 1   b      1
# 2   d      2

print(pd.merge(df1, df2, on='key')) # merge하려면 기준 key가 있어야함. on 부분은 생략됨(가능)
print(pd.merge(df1, df2, on='key', how='inner')) # how도 생략됨 
# 'key' 열을 기준으로 병합(inner join, 교집합)
# 양쪽에 공통적으로 있는 요소만 출력
#    data1 key  data2
# 0      0   b      1
# 1      1   b      1
# 2      2   b      1
# 3      4   a      0
# 4      5   a      0
# 5      6   b      1

print(pd.merge(df1, df2, on='key', how='outer')) # 값 없는 녀석은 결측치가 됨
#    data1 key  data2
# 0    4.0   a    0.0
# 1    5.0   a    0.0
# 2    0.0   b    1.0
# 3    1.0   b    1.0
# 4    2.0   b    1.0
# 5    6.0   b    1.0
# 6    3.0   c    NaN
# 7    NaN   d    2.0

print(pd.merge(df1, df2, on='key', how='left')) # left,right는 기본 outer
print(pd.merge(df1, df2, on='key', how='right'))

'''
공통 칼럼이 없는 경우!!
'''
df3 = pd.DataFrame({'key2':['a','b','c'], 'data2':range(3)}) # df1하고 공통 칼럼 없음
#   key2  data2
# 0    a      0
# 1    b      1
# 2    c      2

print(pd.merge(df1,df3, left_on='key', right_on='key2')) # merge 여건 마련
#    data1 key key2  data2
# 0      0   b    b      1
# 1      1   b    b      1
# 2      2   b    b      1
# 3      3   c    c      2
# 4      4   a    a      0
# 5      5   a    a      0
# 6      6   b    b      1

print(pd.concat([df1, df3], axis=1)) # 0 열단위 결함, 1 행단위 결합
#    data1 key key2  data2
# 0      0   b    a    0.0
# 1      1   b    b    1.0
# 2      2   b    c    2.0
# 3      3   c  NaN    NaN
# 4      4   a  NaN    NaN
# 5      5   a  NaN    NaN
# 6      6   b  NaN    NaN

# series에도 concat 가능
s1 = pd.Series([0,1], index=['a','b'])
s2 = pd.Series([2,3,4], index=['c','d','e'])
s3 = pd.Series([5,6], index=['f','g'])
print(pd.concat([s1, s2, s3], axis=0))
# a    0
# b    1
# c    2
# d    3
# e    4
# f    5
# g    6

'''
그룹화 : pivot_table
'''
data ={'city':['강남','강북','강남','강북'],
       'year':[2000,2001,2002,2003],
       'pop':[3.3, 2.5, 3.0,2.0],             
       }
df = pd.DataFrame(data)
#   city  year  pop
# 0   강남  2000  3.3
# 1   강북  2001  2.5
# 2   강남  2002  3.0
# 3   강북  2003  2.0

print(df.pivot(index='city', columns='year', values='pop'))
print(df.set_index(['city','year']).unstack())
# year  2000  2001  2002  2003
# city
# 강남     3.3   NaN   3.0   NaN
# 강북     NaN   2.5   NaN   2.0

# pivot과 groupby의 중간적 성격
print(df.pivot_table(index=['city']))
print(df.pivot_table(index=['city'], aggfunc='mean')) # aggfunc='mean' 기본값(생략 가능)
#        pop    year
# city
# 강남    3.15  2001.0
# 강북    2.25  2002.0

print(df.pivot_table(index=['city','year'], aggfunc=[len,'sum']))
#           len  sum
#           pop  pop
# city year
# 강남   2000   1  3.3
#      2002   1  3.0
# 강북   2001   1  2.5
#      2003   1  2.0

print(df.pivot_table(index='city',values='pop', aggfunc='mean'))
#        pop
# city
# 강남    3.15
# 강북    2.25

print(df.pivot_table(values=['pop'], index=['year'],columns=['city'],margins=True, fill_value=0))
# margins = True : 총계 추가
# margins_name = : 총계의 행과 열에 붙일 이름
#        pop
# city    강남    강북  All
# year
# 2000  3.30  0.00  3.3
# 2001  0.00  2.50  2.5
# 2002  3.00  0.00  3.0
# 2003  0.00  2.00  2.0
# All   3.15  2.25  2.7

hap = df.groupby(['city'])
print(hap)
print(hap.sum())
print(df.groupby(['city']).sum())
#       year  pop
# city
# 강남    4002  6.3
# 강북    4004  4.5

print(df.groupby(['city','year']).mean())
#            pop
# city year
# 강남   2000  3.3
#      2002  3.0
# 강북   2001  2.5
#      2003  2.0







