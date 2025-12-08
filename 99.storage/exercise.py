import MySQLdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import sys

def main():
    CONFIG = MySQLdb.connect(
    host='127.0.0.1',
    user='root',
    password='12345',
    database='mydb',
    port=3306,
    charset='utf8mb4'
)
    '''
    MariaDb에 저장된 jikwon 테이블을 사용한다.
    담당 고객이 없는 직원의 수, 연봉 평균, 중앙값, 표준편차를 출력하고 이들에 대한 연봉 관련 히스토그램을 출력하는 프로그램을 작성하시오.
    조건 :
    1) DB의 자료를 SQL문으로 읽어, DataFrame에 저장한다.
    2) 아래의 main() 함수에 적당한 코드를 완성하시오. (배점:10)
    '''

    try:
        with CONFIG:       
            cursor = CONFIG.cursor()
            sql = """
                select j.jikwonno, j.jikwonname, j.jikwonpay, g.gogekdamsano
                from jikwon as j left join gogek as g
                on j.jikwonno=g.gogekdamsano
            """
            cursor.execute(sql)            
            df = pd.DataFrame(cursor.fetchall(), columns=['jikwonno', 'jikwonname', 'jikwonpay','gogekdamsano'])          
            print(df)       
            plt.hist(df, bins=20)
            plt.show()

    except Exception as e:
        print(f'에러 : {e}')
    


    
   
if __name__ == "__main__":
    main()