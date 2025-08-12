from django.shortcuts import render
from django.db import connection
import pandas as pd
from datetime import datetime
from django.conf import settings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')  # 윈도우: 'Malgun Gothic', Mac: applegothic
plt.rcParams['axes.unicode_minus'] = False     # 마이너스(-) 깨짐 방지

# Create your views here.
def index(request):
    return render(request, 'index.html')

def showdata(request):
    sql = """
        SELECT j.jikwonno as 사번, j.jikwonname as 직원명, b.busername as 부서명, j.jikwonjik as 직급, j.jikwonpay as 연봉, j.jikwonibsail as 근무년수, b.buserno as 부서번호
        from jikwon as j inner join buser as b
        on j.busernum = b.buserno
    """
    with connection.cursor() as cur:
        cur.execute(sql)

        rows = cur.fetchall()
        cols = [c[0] for c in cur.description]
        # print(cols)
    
    # 1번 DataFrame에 저장 
    df = pd.DataFrame(rows, columns=cols)
    df_sort = df.sort_values(by=['부서번호','직원명'], ascending=False)
    today = datetime.now().year
    # print(today)
    df_sort['근무년수'] = pd.to_datetime(df_sort['근무년수'])
    df_sort['근무년수'] = today - df_sort['근무년수'].dt.year
    # print(df_sort.head())
    sorted_df = df_sort.to_html(classes='table table-stripped', index=False)

    # 2번
    df2 = pd.pivot_table(df_sort, index=['부서명','직급'],values='연봉', aggfunc=['sum','mean'])
    # print(df2)

    # 3번
    static_app_dir = Path(settings.BASE_DIR) / 'static' / 'images'
    static_app_dir.mkdir(parents=True, exist_ok=True, ) # 부모창 밑에다 만든다, 존재하면 pass 없으면 mkdir
    img_path = static_app_dir/'df3.png'

    df3 = pd.pivot_table(df_sort, index=['부서명'],values='연봉', aggfunc=['sum','mean'])

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.4
    x = range(len(df3.index))
    print(len(df3.index))
    ax.bar([i - bar_width/2 for i in x], df3[('sum', '연봉')], width=bar_width, label='연봉 합계')
    ax.bar([i + bar_width/2 for i in x], df3[('mean', '연봉')], width=bar_width, label='연봉 평균')

    ax.set_xticks(x)
    ax.set_xticklabels(df3.index)
    ax.set_ylabel('연봉')
    ax.set_title('부서별 연봉 합계 및 평균')
    ax.legend()
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
     
    # table_html = df3.to_html(classes='table table-stripped', index=False)

    # 4번
    sql4 = """
        SELECT jikwongen as 성별, jikwonjik as 직급
        from jikwon
    """
    with connection.cursor() as cur:
        cur.execute(sql4)
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description]

    df4 = pd.DataFrame(rows, columns=cols)
          
    df4_ct = pd.crosstab(df4['성별'],df4['직급'], margins=True).reset_index()
    # print(df4_ct)
    df4_html = df4_ct.to_html(index=False)


    context_dict = {
        'img_showpath':'images/df3.png',
        'df4_html': df4_html,
        'sorted_df':sorted_df
        
    }

    return render(request, 'showdata.html', context_dict)
    # return render(request, 'showdata.html',context_dict)

'''
   1) 사번, 직원명, 부서명, 직급, 연봉, 근무년수를 DataFrame에 기억 후 출력하시오. (join)
       : 부서번호, 직원명 순으로 오름 차순 정렬 
   2) 부서명, 직급 자료를 이용하여  각각 연봉합, 연봉평균을 구하시오.
   3) 부서명별 연봉합, 평균을 이용하여 세로막대 그래프(plt.bar)를 출력하시오.
   4) 성별, 직급별 빈도표를 출력하시오.
'''