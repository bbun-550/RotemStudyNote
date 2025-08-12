from django.shortcuts import render
from django.db import connection
from django.utils.html import escape
import pandas as pd



# Create your views here.
def indexFunc(request):
    return render(request, 'index.html')

def dbshow(request):
    # 부서명 받는다
    dept = request.GET.get('dept','').strip()

    # inner join
    sql = """
        SELECT j.jikwonno as 직원번호, j.jikwonname as 직원명, b.busername as 부서명,
        b.busertel as 부서전화,j.jikwonpay as 연봉,j.jikwonjik as 직급
        from jikwon as j inner join buser as b
        on j.busernum = b.buserno
    """

    params = []
    if dept: # 부서명이 있으면
        sql += " where b.busername like %s"
        params.append(f'%{dept}%') # mapping 방식으로 sql injection 안걸릴 수 있음. SQL 해킹방지(secure code)
    sql+= " order by j.jikwonno" # 띄어쓰긱 꼭 해줘라

    with connection.cursor() as cur:
        cur.execute(sql, params)

        rows = cur.fetchall() # 전부 불러오기
        # print(f'{cur.description}')
        cols = [c[0] for c in cur.description] # 칼럼명만 슬라이싱
        # print(f'cols : {cols}') # ['직원번호', '직원명', '부서명', '부서전화', '연봉', '직급']
    
    df = pd.DataFrame(rows, columns=cols)
        # print(df.head(3))
    
    # join 결과로 html 생성
    if not df.empty:
        join_html = df[['직원번호','직원명','부서명','부서전화','연봉','직급']].to_html(index=False) # 다 넣을 필요없음
    else:
        join_html = '조회된 자료 없습니다'
    
    # 직급별 연봉 통계표 (NaN은 0으로)
    if not df.empty:
        stats_df = df.groupby('직급')['연봉'].agg(평균='mean', 표준편차=lambda x:x.std(ddof=0), 인원수='count').round(2).reset_index().sort_values(by='평균', ascending=False)
        # ddof : 자유도

        stats_df['표준편차'] = stats_df['표준편차'].fillna(0)
        stats_html = stats_df.to_html(index=False)
    else:
        stats_df = '통계 대상 자료 없습니다'


    # dict로 보낼거야
    ctx_dict = { # 해킹 방지
        'dept':escape(dept), # escape 문자열에 특수문자가 있는 경우, html entity로 치환(단순문자취급)
                # eg. escape('<script>alert(1)</script>') -> '&lt;script&lt; ...' 로 바꿔줌
        'join_html' : join_html,
        'stats_html' : stats_html,
    }

    return render(request, 'dbshow.html', ctx_dict)