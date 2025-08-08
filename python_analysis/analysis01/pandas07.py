'''
웹문서 읽기
# 위키백과 문서 읽기(이순신 자료)
'''
import urllib.request as req
from bs4 import BeautifulSoup
import urllib

'''
url = "https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%88%9C%EC%8B%A0"
wiki = req.urlopen(url)
# print(wiki)
# <http.client.HTTPResponse object at 0x000001B58FBC2740>

soup = BeautifulSoup(wiki, 'html.parser')
print(soup.select('#mw-content-text > div.mw-content-ltr.mw-parser-output > p')) # select 대신 find 써도 됨
'''

# 네이버 제공 코스피 정보 읽기 - DataFrame에 담아서 작업
url_template = 'https://finance.naver.com/sise/sise_market_sum.naver?&page={}'
csv_frame = 'naverkospi.csv'

import csv
import re
import pandas as pd
import requests

'''
with open(csv_frame, mode='w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    # 제목 표시
    headers = 'N	종목명	현재가	전일비	등락률	액면가	시가총액  상장주식수	외국인비율	거래량	PER	ROE	토론'.split()
    # ['N','종목명','현재가','전일비','등락률','액면가','시가총액','상장주식수','외국인비율','거래량','PER','ROE','토론']
    writer.writerow(headers) 

    for page in range(1, 11): # 여러 페이지 가져오기
        url = url_template.format(page)        
        # https://finance.naver.com/sise/sise_market_sum.naver?&page=1
        # https://finance.naver.com/sise/sise_market_sum.naver?&page=2

        res = requests.get(url)
        res.raise_for_status() # 실패하면 작업 중지 아니면 계속
        soup = BeautifulSoup(res.text, 'html.parser') # html 안되면 xml
        # rows = soup.find('table', attrs={'class':'type_2'}).find('tbody').find_all('tr')
        # css
        rows = soup.select('table.type_2 tbody tr')
        
        for row in rows:
            cols = row.find_all('td')
            # if all(cell == '' for cell in row_data):
            #     continue
            if len(cols) < len(headers): # headers 개수 만큼 columns가져오기
                # print(f'[스킵] 열 수 부족: {len(cols)}개')
                continue
            row_data = [re.sub(r'[\n\t]+','', col.get_text().strip()) for col in cols] # sub : 대체한다
            writer.writerow(row_data)

print('csv 저장 성공?')
'''
df = pd.read_csv(csv_frame, dtype=str, index_col=False) # index_col : 첫번째 열을 인덱스로 쓰지마

print(df.columns.to_list()) # 칼럼명만 볼 수 있음
# ['N', '종목명', '현재가', '전일비', '등락률', '액면가', '시가총액', '상장주식수', '외국인비율', '거래량', 'PER', 'ROE', '토론']

print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 500 entries, 0 to 499
# Data columns (total 13 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   N       500 non-null    object
#  1   종목명     500 non-null    object
#  2   현재가     500 non-null    object
#  3   전일비     500 non-null    object
#  4   등락률     500 non-null    object
#  5   액면가     500 non-null    object
#  6   시가총액    500 non-null    object
#  7   상장주식수   500 non-null    object
#  8   외국인비율   500 non-null    object
#  9   거래량     500 non-null    object
#  10  PER     368 non-null    object
#  11  ROE     350 non-null    object
#  12  토론      0 non-null      object
# dtypes: object(13)
# memory usage: 50.9+ KB
# None

numeric_cols = ['현재가','전일비','등락률', '액면가', '시가총액', '상장주식수', '외국인비율', '거래량', 'PER', 'ROE']

# 전일비 전용 전처리 함수
def clean_change_direction(val):
    if pd.isna(val):
        return pd.NA
    val = str(val)
    val = val.replace(',','').replace('상승','+').replace('하락','-')
    val = re.sub(r'[^\d\.\-\+]','',val) # 숫자/기호 외 문자 제거
    try:
        return float(val)
    except ValueError:
        return pd.NA
    
df['전일비'] = df['전일비'].apply(clean_change_direction)
# 0  1      삼성전자   68,800 -1100.0

# 일반 숫자형 커럼 전처리
def clean_numeric_columns(series):
    return (
        series.astype(str)
            .str.replace(',','', regex=False) # regex :정규표현식이야
            .str.replace('%','', regex=False)
            .replace(['','-','N/A','nan'], pd.NA) # 처리할 수 없는 거는 NaN 처리하라
            .apply(lambda x:pd.to_numeric(x, errors='coerce'))
    )

for col in numeric_cols:
    df[col] = clean_numeric_columns(df[col])

# print(f'숫자 칼럼 일괄 처리 후\n{df.head()}')
#    N        종목명      현재가      전일비   등락률  ...  외국인비율       거래량    PER    ROE   토론
# 0  1       삼성전자    68800  -1100.0 -1.57  ...  50.56  12542410  13.33   9.03  NaN
# 1  2     SK하이닉스   258500  -5000.0 -1.90  ...  55.12   1374559   7.24  31.06  NaN
# 2  3   LG에너지솔루션   384000  -2500.0 -0.65  ...   4.19    301979 -77.20  -4.93  NaN
# 3  4   삼성바이오로직스  1031000 -20000.0 -1.90  ...  12.90     43461  57.35  10.45  NaN
# 4  5  한화에어로스페이스   936000 -25000.0 -2.60  ...  44.15    184646  18.93  53.94  NaN

print(df.describe())
print(df[['종목명','현재가','전일비']].head(3))
#         종목명     현재가     전일비
# 0      삼성전자   68800 -1100.0
# 1    SK하이닉스  258500 -5000.0
# 2  LG에너지솔루션  384000 -2500.0

# 시가총액 top 5
top5 = df.dropna(subset=['시가총액']).sort_values(by='시가총액', ascending=False).head(5)
print(top5[['종목명','시가총액']])
#          종목명     시가총액
# 0       삼성전자  4072711
# 1     SK하이닉스  1881886
# 2   LG에너지솔루션   898560
# 3   삼성바이오로직스   733804
# 4  한화에어로스페이스   482633



