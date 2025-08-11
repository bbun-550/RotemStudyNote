from django.shortcuts import render
import json, os # dump, load 활용
import pandas as pd
import numpy as np
import requests
from django.conf import settings
from datetime import datetime

# 폴더 만들기
DATA_DIR = os.path.join(settings.BASE_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'seatle_weather.csv')
CSV_URL = 'https://raw.githubusercontent.com/vega/vega-datasets/master/data/seattle-weather.csv'

# Create your views here.
def index(request):
    return render(request, 'index.html')

# csv 데이터 파일이 없으면 다운로드해서 저장하는 함수
def csvFunc():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        res = requests.get(CSV_URL, timeout=20) # 20초 기다려보겠다
        res.raise_for_status() # HTTP 상태 코드 200(성공)이 아니면 예외를 발생시킴

        # 저장
        with open(CSV_PATH, mode='wb') as f: # binary로 저장해야 메모리를 적게 차지
            f.write(res.content) # res.text 할 수 있으나 한글이 깨질 수 있음. 한글 안정적으로 처리하기 위해 .content



def show(request):
    csvFunc() # 자료가 없으면 url로 가서 파일로 저장. 데이터 확보
    df = pd.read_csv(CSV_PATH)
    # print(df.columns) # ['date', 'precipitation', 'temp_max', 'temp_min', 'wind', 'weather']
    # df.info() # 결측치 확인

    # 일부 열만 참여
    df = df[['date', 'precipitation', 'temp_max', 'temp_min']].copy() # 원본 유지 사본 생성
    df['date'] = pd.to_datetime(df['date']) # type 바꿈. 날짜 연산이 가능해짐
    df = df.dropna() # na가 있는 행은 drop

    # 기술 통계 : 평균, 표준편차, 최소값, 최빈값...
    stats_df = df[['precipitation', 'temp_max', 'temp_min']].describe().round(3)
    # print(f'stats_df : {stats_df}')

    # df의 상위 5행만 출력에 참여
    head_html = df.head().to_html(classes='table table-sm table-striped', index=False, border=0)
    stats_html = stats_df.to_html(classes='table table-sm table-striped', index=False, border=0)

    # Echarts용 데이터 (월별 평균 최고 기온) -> 라인차트 그릴 예정
    # 월 단위 평균 최고 기온 집계
    monthly = (
        df.set_index('date').resample('ME')[['temp_max']] # ME : month end 각 달의 마지막 날 기준. 일정 단위로 집계
        .mean()
        .reset_index() # 구조 원상복귀
    )
    # print(monthly.head(2))
    # 2012-01-31   7.054839 -> 2012  7.05 로 바꾸고 싶음
    labels = monthly['date'].dt.strftime('%Y-%m').tolist()
    # print(labels)

    series = monthly['temp_max'].round(2).tolist()
    # print(series)

    # 웹상에서 리스트 안넘어감 -> json으로 바꿔줘야함
    context_dict = { # 한 번에 여러개 가져가기
        'head_html':head_html,
        'stats_html':stats_html,
        'labels_json':json.dumps(labels, ensure_ascii=False), # ensure_ascii=False : 원 데이터가 넘어갈 수 있게(노파심) . list -> json
        'series_json':json.dumps(series,ensure_ascii=False),
    }

    return render(request, 'show.html', context_dict)


