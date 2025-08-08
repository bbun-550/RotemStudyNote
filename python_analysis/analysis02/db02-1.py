'''
pandas 문제 7)
 a) MariaDB에 저장된 jikwon, buser, gogek 테이블을 이용하여 아래의 문제에 답하시오.
    - 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
    - DataFrame의 자료를 파일로 저장
    - 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
    - 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
    - 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시
    - 부서명별 연봉의 평균으로 가로 막대 그래프를 작성
'''
import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

conn = MySQLdb.connect( # connect는 아래와 같은 형식을 입력받는다
    host='127.0.0.1',
    user='root',
    password='12345',
    database='mydb',
    port=3306,
    charset='utf8'
)

try:    
    cursor = conn.cursor()

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
    print('부서별 연봉의 합 :', df11.sum())

    ctab = pd.crosstab(df['busername'],df['jikwonjik'], margins=True)
    print(ctab)
    print('='*50)
    sql_gogek = """
        select jikwonname, jikwonno, gogekname, gogektel, gogekjumin, gogekdamsano
        from jikwon inner join gogek
        on jikwon.jikwonno=gogek.gogekdamsano
    """
    cursor.execute(sql_gogek)
    for (a,b,c,d,e,f) in cursor:
        print(a,b,c,d,e,f)

    print(df11.mean())
    plt.barh(df11.mean(), labels=df11)
    plt.show()

except Exception as e:
    print(f'문제 발생 : {e}')

finally:
    cursor.close()
    conn.close()




'''
직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시
부서명별 연봉의 평균으로 가로 막대 그래프를 작성
'''



'''
 b) MariaDB에 저장된 jikwon 테이블을 이용하여 아래의 문제에 답하시오.
     - pivot_table을 사용하여 성별 연봉의 평균을 출력
     - 성별(남, 여) 연봉의 평균으로 시각화 - 세로 막대 그래프
     - 부서명, 성별로 교차 테이블을 작성 (crosstab(부서, 성별))
'''




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