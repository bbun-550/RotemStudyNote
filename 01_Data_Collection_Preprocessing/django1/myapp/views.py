from django.shortcuts import render
from django.conf import settings
from pathlib import Path
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # matplotlib 그래프를 그릴 때 backend로 지정하는 코드
# use('Agg') 또는 use('pdf') ... -> GUI없이 시각화를 파일로 저장
# plt.show()해서 창에 보여짐
# 웹에서 출력. 이미지를 띄울거임. gui 환경없이
import matplotlib.pyplot as plt



# Create your views here.
def main(request):
    return render(request, 'main.html')


def showdata(request):
    # 먼저 데이터 로딩
    df = sns.load_dataset('iris')

    # 이미지 저장 경로를 설정 <BASE_DIR?/static/images/>
    static_app_dir = Path(settings.BASE_DIR)/'static'/'images'
    static_app_dir.mkdir(parents=True, exist_ok=True, ) # 부모창 밑에다 만든다, 존재하면 pass 없으면 mkdir
    img_path = static_app_dir/'iris.png'

    # 차트 파일로 저장할거야
    # pie, 꽃의 종류별 개수count
    counts = df["species"].value_counts().sort_index()
    print(f'counts: {counts}')
    plt.figure()
    counts.plot.pie(autopct='%1.1f%%', startangle=90,ylabel='') # percent 출력 소수첫째자리 eg. 35.1%
    # startangle : 시작각도 지정
    plt.title('iris Speicals count')
    plt.axis('equal') # x축, y축 맞추기
    plt.tight_layout()
    plt.savefig(img_path) # img_path 경로에 저장
    plt.savefig(img_path, dpi=130) # 이미지 해상도. 인치 당 점 몇개 찍을꺼야? 숫자가 클수록 선명
    plt.close() # 데이터 누수 방지

    # df를 데이블 태그로 만들어서 show.html에 전달
    table_html = df.to_html(classes='table table-stripped', index=False) # 테이블 꾸미기
    # ctab, pivot_table 다 to_html할 수 있는 것은 아니다. 확인 필요

    return render(request, 'show.html', 
                  {'table':table_html,
                   'img_relpath':'images/iris.png'
                  })




