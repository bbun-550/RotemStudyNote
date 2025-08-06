'''
file i/o    
'''
import pandas as pd

df = pd.read_csv('./ex01.csv') # . 현재경로라는 뜻(상대경로, 절대경로 입력가능)
df = pd.read_csv('./ex01.csv', sep=',') # sep가 생략되어 있음. 
df = pd.read_table('./ex01.csv', sep=',')
#    bunho irum  kor  eng
# 0      1  홍길동   90   90
# 1      2  신기해   95   80
# 2      3  한국인  100   85

print(df, type(df))
print(df.info())
# RangeIndex: 3 entries, 0 to 2
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   bunho   3 non-null      int64
#  1   irum    3 non-null      object
#  2   kor     3 non-null      int64
#  3   eng     3 non-null      int64
# dtypes: int64(3), object(1)
# memory usage: 228.0+ bytes
# None


df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv')
#    1   2   3   4  hello
# 0  5   6   7   8  world
# 1  9  10  11  12    foo

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv',
                 header=None)
#    0   1   2   3      4
# 0  1   2   3   4  hello
# 1  5   6   7   8  world
# 2  9  10  11  12    foo

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv',
                 header=None, names=['a','b','c','d','msg']) # 열 이름 오른쪽부터 채워나감
#    a   b   c   d    msg
# 0  1   2   3   4  hello
# 1  5   6   7   8  world
# 2  9  10  11  12    foo

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv',
                 header=None, names=['a','b','c','d','msg'], skiprows=1) # 열 생략
#    a   b   c   d    msg
# 0  5   6   7   8  world
# 1  9  10  11  12    foo

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt',
                 )
print(df.info()) # read_csv 했지만 대상 파일이 txt(공백으로 구분되어 있음) 그래서 data 하나로 인식
# RangeIndex: 4 entries, 0 to 3
# Data columns (total 1 columns):
#  #   Column                             Non-Null Count  Dtype
# ---  ------                             --------------  -----
#  0               A         B         C  4 non-null      object
# dtypes: object(1)

df = pd.read_table('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt',
                 sep='\s+')
# Index: 4 entries, aaa to ddd
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   A       4 non-null      float64
#  1   B       4 non-null      float64
#  2   C       4 non-null      float64
# dtypes: float64(3)
# memory usage: 128.0+ bytes
# None

df = pd.read_fwf('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/data_fwt.txt',
                 widths=(10,3,5), names=('data','name','price'), encoding='utf-8') # 폭이 일정할 때 fwf 씀
#          data name  price
# 0  2017-04-10  네이버  32000
# 1  2017-04-11  네이버  34000
# 2  2017-04-12  네이버  33000
# 3  2017-04-10  코리아  22000
# 4  2017-04-11  코리아  21000
# 5  2017-04-12  코리아  24000

# url = 'https://ko.wikipedia.org/wiki/%EB%A6%AC%EB%88%85%EC%8A%A4'
# df = pd.read_html(url)
# print(f'총 {len(df)}개 자료')

'''
청크 Chunk : 대량의 데이터 파일 읽는 경우
chunk 단위로 분리해서 읽기 가능
1) 메모리 절약 가능
2) 스트리밍 방식으로 순차적으로 처리할 수 있음(로그 분석 ; 시스템 데이터, 실시간 데이터, ML 데이터 처리)
3) 분산 처리(batch 묶어서 처리)
4) 속도는 느림
순차적 sequential
'''
import time
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='malgun gothic') # 한글 깨짐 방지

n_rows = 10000
''' 빅데이터 파일 준비
data = {
    'id':range(1,n_rows+1),
    'name':[f'student_{i}' for i in range(1,n_rows+1)],
    'score':np.random.randint(50,101,size=n_rows),
    'score2':np.random.randint(50,101,size=n_rows)
}
df = pd.DataFrame(data)

csv_path = 'students.csv'
df.to_csv(csv_path, index=False) # 파일 저장, 인덱스 빼고 저장
'''


# 작성된 csv 파일 사용 : 전체 한 방에 처리
start_all = time.time() # 현재 시간
df_all = pd.read_csv('./students.csv')

average_all1 = df_all['score'].mean() # 한 방에 처리하는 거라 all이름 붙임
average_all12 = df_all['score2'].mean()
time_all = time.time() - start_all
print(time_all)
# 소요시간 0.0055828094482421875

# chunk로 처리
chunk_size = 1000 # 데이터 1000개 처리
total_score1 = 0
total_score2 = 0
total_count = 0

start_chunk_total = time.time()
for i, chunk in enumerate(pd.read_csv('./students.csv', chunksize=chunk_size)):
    start_chunk = time.time() # chunk 단위로 시간 확인
    # 청크 처리할 때마다 첫 번째학생 정보만 출력
    first_student = chunk.iloc[0]
    print(f'Chunk {i+1} 첫 번째 학생: id={first_student['id']}, 이름={first_student['name']},'
          f'scorce1={first_student['score']}, scorce2={first_student['score2']}')
    total_score1 += chunk['score'].sum()
    total_score2 += chunk['score2'].sum()
    total_count += len(chunk)
    end_chunk = time.time()
    elapsed = end_chunk - start_chunk
    print(f'         처리 시간 : {elapsed}초')
# Chunk 1 첫 번째 학생: id=1, 이름=student_1,scorce1=74, scorce2=66
#          처리 시간 : 0.00036406517028808594초
# Chunk 2 첫 번째 학생: id=1001, 이름=student_1001,scorce1=65, scorce2=52
#          처리 시간 : 0.00027632713317871094초
# ...
# Chunk 9 첫 번째 학생: id=8001, 이름=student_8001,scorce1=88, scorce2=82
#          처리 시간 : 0.0002739429473876953초
# Chunk 10 첫 번째 학생: id=9001, 이름=student_9001,scorce1=98, scorce2=74
#          처리 시간 : 0.00027179718017578125초


time_chunk_total = time.time() - start_chunk_total
average_chunk_1 = total_score1 / total_count # score 전체 평균
average_chunk_2 = total_score2 / total_count # score2 전체 평균

print(f'---처리 결과----\n전체 학생 수: {total_count}\nscore1 총합: {total_score1}, 평균:{average_chunk_1:.2f}\n'
      f'score2 총합: {total_score2}, 평균:{average_chunk_2:.2f}\n'
      f'전체 한 번에 처리한 경우 소요시간: {time_all:.4f}초\n'
      f'청크로 처리한 경우 소요시간: {time_chunk_total:.4f}초'
      )
# ---처리 결과----
# 전체 학생 수: 10000
# score1 총합: 747322, 평균:74.73
# score2 총합: 748435, 평균:74.84
# 전체 한 번에 처리한 경우 소요시간: 0.0059초
# 청크로 처리한 경우 소요시간: 0.0303초

# 시각화 맛보기
labels = ['전체 한 번에 처리','청크로 처리']
times = [time_all,time_chunk_total]
plt.figure(figsize=(6,4)) # 크기
bars = plt.bar(labels, times, color=['skyblue','yellow']) # 막대 그래프

for bar, time_val in zip(bars, times): # zip 튜플로 묶어주는 역할
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
             f'{time_val:.4f}초', ha='center', va='bottom', # ha, va : 어디에 표시할꺼야 
             fontsize=10  ) 
plt.ylabel('처리시간(초)')
plt.title('전체 한번에 처리 vs 청크로 처리')
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()



