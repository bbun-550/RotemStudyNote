'''
카이제곱 검정

카이제곱 문제2) 지금껏 A회사의 직급과 연봉은 관련이 없다.
그렇다면 jikwon_jik과 jikwon_pay 간의 관련성 여부를 통계적으로 가설검정하시오.
  예제파일 : MariaDB의 jikwon table 
  jikwon_jik   (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
  jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
  조건 : NA가 있는 행은 제외한다.
'''
# pip install mysqlclient
# import pymysql
# pymysql.install_as_MySQLdb()
import MySQLdb
import numpy as np
import pandas as pd
import scipy.stats as stats


# 가설
# 귀무가설 : 직급과 연봉 간의 연관성이 없다
# 대립가설 : 직급과 연봉 간의 연관성이 있다.

conn = MySQLdb.connect(
    host='127.0.0.1',
    user='root',
    password='12345',
    database='mydb',
    port=3306,
    charset='utf8'
)

try:
    with conn:
        cursor = conn.cursor()

        sql="""
            select jikwonjik,jikwonpay 
            from jikwon
        """
        cursor.execute(sql)

        df = pd.DataFrame(cursor.fetchall(),
                          columns=['직급','연봉'])
        
        bins = np.arange(1000,10000,2000)
        df['범위'] = pd.cut(df['연봉'], bins=bins, right=True)
        df = df.replace({'이사':1, '부장':2, '과장':3, '대리':4, '사원':5})
        df = df.dropna().reset_index(drop=True)
        print(df)

        df_ctab = pd.crosstab(index=df['직급'], columns=df['범위'])
        print(df_ctab)

        chi2, pvalue, dof, _ = stats.chi2_contingency(df_ctab)
        print(f'chi2 : {chi2:.4f}\npvalue : {pvalue:.10f}\ndof : {dof}')
        # 1 - 이사 포함 결과
        # chi2 : 65.5604
        # pvalue : 0.0000000590
        # dof : 16

        # 1 - 이사 제외 결과
        # chi2 : 34.3751
        # pvalue : 0.0000767602
        # dof : 9
        
        # 결론 : p-value(0.0000000590) < 유의수준 α(0.05) 이므로 귀무가설을 기각한다
        # 직급과 연봉 간의 높은 연관성을 보인다. 데이터 결과는 우연하게 일어난 것이 아니라 어떠한 원인에 의해서 얻어진 값이다.
        # 어떠한 원인이란?
        
except Exception as e:
    print(f'오류!! : {e}')
