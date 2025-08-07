'''
웹 스크랩핑
'''
from bs4 import BeautifulSoup
import urllib.request # 실무용은 아니고 연습용(코드를 장황하게 만듦)
import requests # 코드 간결, 실무용
import pandas as pd

url = 'https://www.kyochon.com/menu/chicken.asp'
response = requests.get(url)
response.raise_for_status() # 오류 발생 시 예외 처리

soup = BeautifulSoup(response.text, 'lxml') # lxml.parser는 빠름, html.parser는 기본
# print(soup) # HTML 구조를 보여줌

# 메뉴 이름 추출
names = [tag.text.strip() for tag in soup.select('dl.txt > dt')]
# print(names)

# 가격 추출
prices = [int(tag.text.strip().replace(',', '')) for tag in soup.select('p.money strong')]
# 가격이 문자열로 되어 있으므로 정수로 변환
# print(prices)

df = pd.DataFrame({'상품명': names, '가격': prices})
# print(df.head())

print('가격 평균:', round(df['가격'].mean(), 2))
print(f'가격 평균: {df["가격"].mean():.2f}') # f-string 사용
print('가격 표준편차:', round(df['가격'].std(), 2))