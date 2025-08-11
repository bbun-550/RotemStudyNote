# pip install mysqlclient
# import pymysql
# pymysql.install_as_MySQLdb()
import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import csv
plt.rc('font', family='applegothic')  # 윈도우: 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False     # 마이너스(-) 깨짐 방지

'''
# 연결해보자
conn = MySQLdb.connect( # connect는 아래와 같은 형식을 입력받는다
    host='127.0.0.1',
    user='root',
    password='12345',
    database='mydb',
    port=3306,
    charset='utf8'
)
# 소스 파일에 있으면 노출되어 위험하기 때문에 별도 파일로 관리. pickle 활용!
# pickle : 객체를 파일로 저장할 때 사용
# - dump, load
'''
'''
config = { # pickle로 별도 불러올 수 있게 파일로 만들겠음 (dbconfig.py)
    'host':'127.0.0.1',
    'user':'root',
    'password':'12345',
    'database':'mydb',
    'port':3306,
    'charset':'utf8'
} # dict를 쓰려면 unpack 해줘야 함
'''

try:
    with open('./mymariadb.dat', mode='rb') as obj:
        config = pickle.load(obj) # dbconfig.py의 config를 가져온다
except Exception as e:
    print('읽기 오류 :', e)
    sys.exit() # 오류나면 프로그램 강제 종료


try:    
    with MySQLdb.connect(**config) as conn: # ** : unpacking    
        cursor = conn.cursor()

        sql="""
            select jikwonno, jikwonname, busername,jikwonjik,jikwongen,jikwonpay 
            from jikwon inner join buser
            on jikwon.busernum=buser.buserno
        """
        cursor.execute(sql)

        # 출력 1
        # for (a,b,c,d,e,f) in cursor:
        #     print(a,b,c,d,e,f)
        # for (jikwonno, jikwonname, busername,jikwonjik,jikwongen,jikwonpay) in cursor: # 가독성 좋음. 단, 코드가 길다
        #     print(jikwonno, jikwonname, busername,jikwonjik,jikwongen,jikwonpay)

        # 출력 2 : Dataframe으로 출력(자주 사용)
        df1 = pd.DataFrame(cursor.fetchall(),
                        columns=['jikwonno', 'jikwonname', 'busername','jikwonjik','jikwongen','jikwonpay'])
        print('df1:',df1.head(3))

        # 출력 3 : csv 파일로 출력(자주 사용)
        # import csv 필요
        # with open('jik_data.csv', mode='w',encoding='utf8') as fobj: # close 귀찮아서 with open() 사용
        #     writer = csv.writer(fobj)
        #     for r in cursor:
        #         writer.writerow(r)
        
        # csv파일을 dataFrame 넣어서 읽어보자
        df2 = pd.read_csv('./jik_data.csv', header=None, names=['번호', '이름', '부서이름','직급','성별','연봉'])
        print(df2.head())

        # DB 자료를 pandas의 sql 처리 기능으로 읽기
        df = pd.read_sql(sql, conn)
        df.columns = ['번호', '이름', '부서이름','직급','성별','연봉']
        # print(df.head(3))

        # DB의 자료를 DataFrame으로 읽었으므로 pandas의 기능을 적용해보자
        print('건수 :', len(df))
        print('건수 :', df['이름'].count())
        print('직급별 인원수 :', df['직급'].value_counts())
        print('연봉 평균 :', df.loc[:, '연봉'].mean())

        # cross table 
        ctab = pd.crosstab(df['성별'],df['직급'], margins=True) # 성별, 직급별 건수
        # print(ctab)
        # print(ctab.to_html) # 나중에 django로 보낼 때 유용하게 쓰임

        # 시각화 : 직급별 연봉 평균 - pie 그래프 사용
        jik_ypay = df.groupby(['직급'])['연봉'].mean()
        print(f'직급별 연봉 평균 : {jik_ypay.round(2)}')
        # 직급별 연봉 평균 : 직급
        # 과장    7200.00
        # 대리    5064.29
        # 부장    8466.67
        # 사원    3476.92

        # 6가지 데이터가 파이 그래프 선택
        plt.pie(jik_ypay, explode=(0.2,0,0,0.3,0), labels=jik_ypay.index, shadow=True, labeldistance=0.7, counterclock=False) 
        # labeldistance : 글자 위치
        # shadow=True
        # counterclock=False : 기본이 시계반대 방향
        plt.show()

except Exception as e:
    print('처리 오류 :', e)

    
