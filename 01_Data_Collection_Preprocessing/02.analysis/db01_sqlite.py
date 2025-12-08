'''
Local Database 연동 후 자료를 읽어 Dataframe에 저장
'''
import sqlite3

sql = 'create table if not exists test(product varchar(10), maker varchar(10), weight real, price integer)' 
# 없다면 test 이름으로 생성, 4 칼럼 table

conn = sqlite3.connect(":memory:") # db 연결. :memory 램에만 존재. 'testdb'
# 쌍따옴표 써야함. 따옴표 사이에 지정한 경로에 데이터파일 생선한다는 의미
conn.execute(sql)
conn.commit() # 요청 마무리

stmt = 'insert into test values(?,?,?,?)' # ?로 맵핑. +로 더해주면 secure coding 위배
data1= ('mouse','samsung',12.5,5000) # 튜플, 리스트로 데이터 지정 가능
conn.execute(stmt, data1)

data2= ('mouse2','samsung',15.5,5000)
conn.execute(stmt, data2)

# 복수 개 추가
datas= [('mouse3','lg',22.5,15000),('mouse4','lg',25.5,25000)]
conn.executemany(stmt, datas)

cursor = conn.execute("select * from test")
rows = cursor.fetchall()
# print(rows[0], ' ', rows[1], rows[0][1])
for a in rows:
    print(a)

import pandas as pd
df = pd.DataFrame(rows, columns=['product','maker','weight','price'])
print(df)
# print(df.to_html())

df2 = pd.read_sql("select * from test", conn) # execute > fetch 할 필요없음
print(df2)
#   product    maker  weight  price
# 0   mouse  samsung    12.5   5000
# 1  mouse2  samsung    15.5   5000
# 2  mouse3       lg    22.5  15000
# 3  mouse4       lg    25.5  25000

pdata = {
    'product':['연필','볼펜','지우개'],
    'maker':['동아','펜탈','모나미'],
    'weight':[1.5,5.5,10.0],
    'price':[500,1000,15000]
}

# print(type(pdata)) # <class 'dict'>

frame = pd.DataFrame(pdata) # df 만들기만 함 >> DB에 넣을 차례
# print(frame)
#   product maker  weight  price
# 0      연필    동아     1.5    500
# 1      볼펜    펜탈     5.5   1000
# 2     지우개   모나미    10.0  15000

frame.to_sql("test1", conn, if_exists='append',index=False) # if_exists 존재하면 추가. 저장됨

df3 = pd.read_sql("select product 제품명, maker as 메이커, price 가격, weight as 무게 from test1", conn) # db에 저장된 데이터 들어갔는지 보자
# print(df3)
#   product maker  price  weight
# 0      연필    동아    500     1.5
# 1      볼펜    펜탈   1000     5.5
# 2     지우개   모나미  15000    10.0

cursor.close() # 닫기 까먹지마
conn.close()