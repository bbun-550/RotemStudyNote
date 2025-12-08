'''
기술 통계의 목적은 데이터를 수지, 요약, 정리, 시각화
- 도수분포표 frequency distribution : 데이터를 구간별로 나눠 빈도를 정리한 표
    - 이를 통해 데이터의 분포를 한 눈에 파악할 수 있다 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='applegothic') # malgun gothic
plt.rcParams['axes.unicode_minus'] = False

# Step 1 : 데이터를 읽어서 DataFrame에 저장
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/heightdata.csv')
# print(df.head()) # 키 데이터

# step 2 : 최대값 max, 최소값 min
min_height = df['키'].min()
max_height = df['키'].max()
print(f'최소값 : {min_height}cm\n최대값 : {max_height}cm') # 최소값 : 158cm 최대값 : 191cm

# step 3 : 구간 설정, 계급 설정 cut 활용
bins = np.arange(156,195,5) # 156에서 195까지 5구간 나눈다
print(bins) # [156 161 166 171 176 181 186 191]

df['계급'] = pd.cut(df['키'], bins=bins, right=True, # 계급이라는 새로운 칼럼 만든다 ;right=True 구간의 오른쪽 포함
                  include_lowest=True) #
# print(df.head(2)) # 0  158  (155.999, 161.0]
# print(df.tail(2)) # 49  191  (186.0, 191.0]

# step 4 : 각 계급의 중앙값 median 구하기
# df['계급구간'] = pd.cut(df['키'], bins=bins, right=True, include_lowest=True)
df['계급중앙값'] = df['계급'].apply(lambda x: int((x.left + x.right)/2))
# print(df.head())

# step 5 : 도수 계산
freq = df['계급'].value_counts().sort_index() #

# step 6 : 상대 도수 계산 - 전체 데이터에 대한 비율
relative_freq = (freq / freq.sum()).round(2)
# print(relative_freq)
# 계급
# 158    0.10
# 163    0.16
# 168    0.20
# 173    0.26
# 178    0.12
# 183    0.10
# 188    0.06

# step 7 : 누적 도수 계산 - 계급별 도수를 누적 cumsum
cum_freq = freq.cumsum()
# print(cum_freq)
# 계급
# 158     5
# 163    13
# 168    23
# 173    36
# 178    42
# 183    47
# 188    50

# step 8 : 도수분포표 작성
dist_table = pd.DataFrame({
    # 출력 예시 : "156 ~ 161" ...
    '계급':[f"{int(interval.left)} ~ {int(interval.right)}" for interval in freq.index],
    # 계급의 중간 값
    '계급값':[((int(interval.left + interval.right)) / 2) for interval in freq.index],
    # 도수
    '도수':freq.values,
    '상대도수':relative_freq.values,
    '누적도수':cum_freq.values,
})

# dist_table = pd.DataFrame({
#     '계급': [f"{int(interval.left)} ~ {int(interval.right)}" for interval in freq.index],
#     '계급중앙값': [(int(interval.left) + int(interval.right)) / 2 for interval in freq.index],
#     '도수': freq.values,
#     '상대도수': relative_freq.values,
#     '누적도수': cum_freq.values,
# })
print(dist_table)


# step 9 : 히스토그램 그리기
plt.figure(figsize=(9,5))
plt.bar(dist_table['계급값'], dist_table['도수'], width=5, 
        color='cornflowerblue', edgecolor='black')
plt.title('학생 50명 키 히스토그램', fontsize=16)
plt.xlabel('키(계급)')
plt.ylabel('도수')
plt.xticks(dist_table['계급값'])
plt.grid(axis='y', linestyle='--',alpha=0.7)
plt.tight_layout()
plt.show()











