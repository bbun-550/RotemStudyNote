import pickle

config = { 
    'host':'127.0.0.1',
    'user':'root',
    'password':'1234',
    'database':'mydb',
    'port':3306,
    'charset':'utf8'
} # dict를 객체 타입으로 바꿀 예정

with open('./mymariadb.dat',mode='wb') as obj: # wb : 바이너리 형태로 저장한다
    pickle.dump(config, obj)