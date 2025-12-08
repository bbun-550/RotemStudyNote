'''
pandas 문제 7)
 a) MariaDB에 저장된 jikwon, buser, gogek 테이블을 이용하여 아래의 문제에 답하시오.
    - 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
    - DataFrame의 자료를 파일로 저장
    - 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
    - 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
    - (help)직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시
    - 부서명별 연봉의 평균으로 가로 막대 그래프를 작성
'''
# import pymysql # 맥북에서 실행시 필요
# pymysql.install_as_MySQLdb() # 맥북에서 실행시 필요

import MySQLdb
import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')  # 윈도우: 'Malgun Gothic', Mac: applegothic
plt.rcParams['axes.unicode_minus'] = False     # 마이너스(-) 깨짐 방지

conn = MySQLdb.connect( # connect는 아래와 같은 형식을 입력받는다
    host='127.0.0.1',
    user='root',
    password='12345',
    database='mydb',
    port=3306,
    charset='utf8mb4'
)

'''
try:    
    cursor = conn.cursor() # cursor 객체를 만들면 sql 명령문을 입력할 수 있다

    sql="""
        select jikwonno, jikwonname, busername,jikwonpay, jikwonjik
        from jikwon inner join buser
        on jikwon.busernum=buser.buserno
    """
    cursor.execute(sql)

    df = pd.DataFrame(cursor.fetchall(),
                      columns=['jikwonno', 'jikwonname', 'busername','jikwonpay','jikwonjik'])

    # print(df.head(3))
    # df.to_csv('jikwon_info.csv', header=None)

    df11 = df.groupby(['busername'])['jikwonpay']
    result = df11.agg(['sum','max','min'])
    print('부서별 연봉의 합 :\n', result)

    ctab = pd.crosstab(df['busername'],df['jikwonjik'], margins=True)
    print(ctab)
    print('='*50)
    sql_gogek = """
        select j.jikwonname, g.gogekno, g.gogekname, g.gogektel
        from jikwon as j left join gogek as g
        on j.jikwonno=g.gogekdamsano
    """
    cursor.execute(sql_gogek)
    gogek_list = []
    for (a, b, c, d) in cursor:
        gogek_list.append([a, b, c, d])
    gogek_df = pd.DataFrame(gogek_list, columns=['직원명', '고객번호', '고객명', '고객 전화번호'])    
    gogek_df['담당고객'] = gogek_df['고객명'].fillna('X')    
    print(gogek_df[['직원명', '담당고객']])
    
    buser_mean = df11.mean()
    print(buser_mean)
    plt.barh(buser_mean.index, buser_mean.values)
    plt.show()

except Exception as e:
    print(f'문제 발생 : {e}')

finally:
    cursor.close()
    conn.close()
'''

'''
 b) MariaDB에 저장된 jikwon 테이블을 이용하여 아래의 문제에 답하시오.
     - pivot_table을 사용하여 성별 연봉의 평균을 출력
     - 성별(남, 여) 연봉의 평균으로 시각화 - 세로 막대 그래프
     - 부서명, 성별로 교차 테이블을 작성 (crosstab(부서, 성별))
'''

try:
    with conn:
        cursor = conn.cursor()
        
        sql="""
            select jikwonno, jikwonname, busername,jikwonpay, jikwonjik, jikwongen
            from jikwon inner join buser
            on jikwon.busernum=buser.buserno       
        """
        cursor.execute(sql)
        
        df_b = pd.DataFrame(cursor.fetchall(), columns=['jikwonno', 'jikwonname', 'busername','jikwonpay','jikwonjik', 'jikwongen'])
        
        # print(df_b)
        pivot_t = df_b.pivot_table(index=['jikwongen'],values=['jikwonpay'],aggfunc='mean')
        print(pivot_t)
        
        # pivot_table 계속 쓰는 방법
        pivot_plot = df_b.pivot_table(index='jikwongen', values='jikwonpay', aggfunc='mean')
        plt.bar(pivot_plot.index, pivot_plot['jikwonpay'].values)
        plt.title('성별 평균 연봉')
        plt.xlabel('성별')
        plt.ylabel('평균 연봉')
        
        # plt.bar(df_b['jikwongen'],df_b['jikwonpay'].mean()) # y값이 전체 데이터의 평균만 계산. 그래서 막대 길이가 같음
        ## pivot_table 이어서 사용 또는 groupby 사용
        ### gender_mean = df_b.groupby('jikwongen')['jikwonpay'].mean()
        
        # plt.title('성별 평균 연봉')
        # plt.xlabel('성별')
        # plt.ylabel('평균 연봉')
        # plt.tight_layout()
        plt.show()
        
        ctab = pd.crosstab(df_b['busername'],df_b['jikwongen'])
        print(ctab)
    
except Exception as e:
    print(f'오류 발생 : {e}')



'''
 c) 키보드로 사번, 직원명을 입력받아 로그인에 성공하면 console에 아래와 같이 출력하시오.
      조건 :  try ~ except MySQLdb.OperationalError as e:      사용
     사번  직원명  부서명   직급  부서전화  성별
     ...
     인원수 : * 명
'''

'''
try:
    cursor = conn.cursor()
    no = int(input('사원번호 입력:')) # 사원번호 입력
    name = input('이름 입력:') # 이름 입력

    # DB에 직접 입력받은 no, name로 검색
    sql_login = """
        select jikwonno, buser.busername, jikwonname, jikwonjik, buser.busertel, jikwongen
        from jikwon inner join buser
        on jikwon.busernum=buser.buserno
        where jikwonname = %s and jikwonno = %s 
    """
    cursor.execute(sql_login, (name,no))
    result = cursor.fetchone()
    if result:
        print('로그인 성공')
        df = pd.DataFrame([result],
                      columns=['사번','직원명', '부서명','직급','부서전화','성별'])
        print(df)

        sql = """
        select count(*) from jikwon
    """
        cursor.execute(sql)
        count = cursor.fetchone()[0]
        print(f'인원수 :{count}명')
    else:
        print('로그인 실패')     

except Exception as e:
    print(f'문제 발생 :{e}')    
except MySQLdb.OperationalError as e:
    print(f'언제뜨나?: {e}')

finally:
    cursor.close()
    conn.close()

'''