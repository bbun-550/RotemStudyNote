from django.shortcuts import render, redirect
import os
from django.conf import settings # 시각화 파일 저장
import matplotlib
matplotlib.use('Agg') # 서버에서 시각화 GUI 없이 저장할 때 사용한다.
import matplotlib.pyplot as plt
plt.rc('font', family="malgun gothic")

from myapp.models import Survey # 클래스 호출한다.
import pandas as pd
import numpy as np
import scipy.stats as stats


# Create your views here.
def surveyMain(request):
    return render(request, "index.html")


def surveyView(request): # 설문지 창
    return render(request, "coffee/coffeesurvey.html")



def surveyProcess(request): 
    insertDataFunc(request)

    rdata = list(Survey.objects.all().values()) # 타입이 안 맞기 때문에 타입 바꿀 필요 있음
    # print(rdata)
    
    # 데이터분석 : 이원 카이제곱 검정
    crossTb1, results, df = analysisFunc(rdata) # 성격이 다르기 때문에 함수를 만든다

    # chart 처리 함수 호출 15:36
    static_img_path = os.path.join(settings.BASE_DIR,'static','images','cobar.png') # 이미지 호출 및 저장 경로
    save_brand_barFunc(df, static_img_path)

    return render(request, 'coffee/result.html', {
        'crossTb1':crossTb1.to_html(),
        'results':results,
        'df': df.to_html(index=False),
        'static_img_path':static_img_path
    })

def insertDataFunc(request):# DB 입력 함수
    if request.method == 'POST': # 메소드가 post일 때만 작업한다
        # print(request.POST.get('gender'),request.POST.get('age'),request.POST.get('co_survey'))
        Survey.objects.create( # (sql문) insert ..
            gender = request.POST.get('gender'),
            age = request.POST.get('age'),
            co_survey = request.POST.get('co_survey'),
        )

def analysisFunc(rdata):
    df = pd.DataFrame(rdata)
    if df.empty: # test할 때 주석처리
        return pd.DataFrame(), '데이터가 없어요', pd.DataFrame() # crossTb1, results, df 이 셋으로 받을거라 df를 앞 뒤로 배치함
    
    '''
    가설
    # 귀무가설 : 성별에 따라 선호하는 커피 브랜드에 차이가 없다.
    # 대립가설 : 성별에 따라 선호하는 커피 브랜드에 차이가 있다.
    '''

    # 결측치 제거
    df = df.dropna(subset=['gender','co_survey'])

    # 남,여 데이터를 더미 데이터로 바꿔줘야 한다.
    # 범주형 데이터에 대해서 더미화(숫자화)
    df['genNum'] = df['gender'].apply(lambda g:1 if g == '남' else 2)
    df['coNum'] = df['co_survey'].apply(
        lambda c:1 if c == '스타벅스' else 2 if c == '커피빈' else 3 if c == '이디아' else 4
    )
    # print('df : ', df)

    # 교차표 작성 ; 이원 카이제곱 검정에 필요
    crossTb1 = pd.crosstab(index=df['gender'], columns=df['co_survey'])
    # print(crossTb1)

    # 표본 부족시 경고 메세지 전달
    if crossTb1.size == 0 or crossTb1.shape[0] < 2 or crossTb1.shape[1] < 2:
        results = "표본 자료가 부족해서 카이제곱 검정 수행 불가!!"
        return crossTb1, results, df
    
    # 본격 카이제곱 검정
    alpha = 0.05 # 유의수준
    statistic, pvalue, ddof, expected = stats.chi2_contingency(crossTb1)

    # 기대 빈도 최소값 체크 (경고용)
    min_expected = expected.min()
    expected_note = ""
    if min_expected < 5:
        expected_note = f"<br><small>* 주의 : 기대 빈도의 최소값이 {min_expected:.2f}로 5 미만이 있어서 카이제곱 가정에 다소 취약합니다.</small>" # 웹으로 출력 

    if pvalue >= alpha:
        results = (
            f'p값이 {pvalue:.5f}이므로 {alpha} 이상이므로'
            f'성별에 따라 선호 커피 브랜드에 차이가 없다 (귀무가설 채택)' # 카이제곱 검정에 대해서 모르는 사람들을 위해 더 쉽게 작성
            )
    else:
        results = (
            f'p값이 {pvalue:.5f}이므로 {alpha} 미만이므로'
            f'성별에 따라 선호 커피 브랜드에 차이가 있다 (대립가설 채택)'
            )
    return crossTb1, results, df # 함수 리턴은 하나만 된다. 여러 개 넘어가는 것은 튜플 하나가 넘어가는 것이다.

def save_brand_barFunc(df, out_path):

    # 브랜드명(x축)
    if df is None or df.empty or 'co_survey' not in df.columns:
        try: # 기존에 있는 것이 있을 수 있기 때문에 사용
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        return False

    # x축 이름으로 표시
    order = ['스타벅스','커피빈','이디아','탐앤탐스']
    brand_counts = df['co_survey'].value_counts().reindex(order, fill_value=0)

    # color는 무지개로 출력 : 막대 색깔 다채롭게 출력하기
    cmap = plt.get_cmap('rainbow')
    n = len(brand_counts)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    # 차트 출력
    fig = plt.figure()
    ax = brand_counts.plot(kind='bar', width=0.6, color=colors, edgecolor='black')
    ax.set_xlabel('커피사')
    ax.set_ylabel('선호 건수')
    ax.set_title('커피 브랜드 선호 건수')
    ax.set_xticklabels(order, rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def surveyShow(request):
    rdata = list(Survey.objects.all().values()) # 타입이 안 맞기 때문에 타입 바꿀 필요 있음
    # print(rdata)
    
    # 데이터분석 : 이원 카이제곱 검정
    crossTb1, results, df = analysisFunc(rdata) # 성격이 다르기 때문에 함수를 만든다

    # chart 처리 함수 호출
    static_img_path = os.path.join(settings.BASE_DIR,'static','images','cobar.png') # 이미지 호출 및 저장 경로
    save_brand_barFunc(df, static_img_path)

    return render(request, 'coffee/result.html', {
        'crossTb1':crossTb1.to_html(),
        'results':results,
        'df': df.to_html(index=False),
        'static_img_path':static_img_path
    })