'''
BBQ 페이지 자료 읽기
- 메뉴 중 후라이드를대사응로 한다
출력
메뉴명 가격 설명

건수: 
가격평균:
표준편차:
최고가격:
최저가격:
기타:
시각화
'''
from bs4 import BeautifulSoup
import requests
import pandas as pd

url = 'https://web.dominos.co.kr/goods/list?dsp_ctgr=C0101'
# with open('url', mode='r', encoding='utf-8') as f:
#     pizza = f.read()
#     # print(pizza)

# soup = BeautifulSoup(pizza, 'html.parser')
# print(soup.prettify())

response = requests.get(url)
response.raise_for_status()
soup = BeautifulSoup(response.text, 'html.parser')
# print(soup.prettify())

names = [tag.text.strip().replace('\n\nNEW','').replace('\nNEW','') for tag in soup.select('div.menu-list div.subject')]
# print(names)

desc = [tag.text.strip() for tag in soup.select('div.menu-list div.hashtag')]
# print(desc)

priceL = soup.select('div.menu-list span.price')
print(priceL)
# priceM = 

# df = pd.DataFrame({'상품명':names, '설명':desc})
# print(df.head())

'''
response = requests.get(url)
response.raise_for_status() # 오류 발생 시 예외 처리

soup = BeautifulSoup(response.text, 'lxml')
# print(soup.prettify())

# 메뉴
names = soup.select('ul.menu_list > li > div.cont > p.tit')
print(names)
'''