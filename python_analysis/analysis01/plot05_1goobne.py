from bs4 import BeautifulSoup
import requests
import pandas as pd

chick_url = 'https://www.goobne.co.kr/menu/menu_list_p?classId=1&classId2={}&itemId=&source=&quickBrId=&quickDeliverySeq=&dlvType='
piz_url = 'https://www.goobne.co.kr/menu/menu_list_p?classId=12&classId2={}&itemId=&source=&quickBrId=&quickDeliverySeq=&dlvType='
side_url = 'https://www.goobne.co.kr/menu/menu_list_p?classId=16&classId2=&itemId=&source=&quickBrId=&quickDeliverySeq=&dlvType='
sauce_url = 'https://www.goobne.co.kr/menu/menu_list_p?classId=18&classId2={}&itemId=&source=&quickBrId=&quickDeliverySeq=&dlvType='


datas = []

# 치킨
for page in [2,7,27]:
    url = chick_url.format(page)
    response1 = requests.get(url)
    response1.raise_for_status()
    soup1 = BeautifulSoup(response1.text, 'html.parser')
    # print(soup1.prettify())
    # # names = soup.select('div.mini-tabcontents div.textbox > h4')
    chick_names=[tag.text.strip() for tag in soup1.select('div.mini-tabcontents div.textbox h4')]
    # print(chick_names)
    # # prices = soup.select('div.tabcontents div.textbox > p')
    chick_prices=[int(tag.text.strip().replace(',','').replace('원 \n\n 해당 가격은 권장 소비자가 입니다.','')) for tag in soup1.select('div.mini-tabcontents div.textbox p.price b')]
    # print(chick_prices)
    for name, price in zip(chick_names, chick_prices):
        datas.append({'상품명': name, '가격': price, '종류':'치킨'})

# 피자
for page in [13,14,15]:
    url = piz_url.format(page)
    response2 = requests.get(url)
    response2.raise_for_status()
    soup2 = BeautifulSoup(response2.text, 'html.parser')
    piz_names=[tag.text.strip() for tag in soup2.select('div.mini-tabcontents div.textbox h4')]
    piz_prices=[int(tag.text.strip().replace(',','').replace('원 \n\n 해당 가격은 권장 소비자가 입니다.','')) for tag in soup2.select('div.mini-tabcontents div.textbox p.price b')]
    for name, price in zip(piz_names, piz_prices):
        datas.append({'상품명': name, '가격': price, '종류':'피자'})

# 사이드
response3 = requests.get(side_url)
response3.raise_for_status()
soup3 = BeautifulSoup(response3.text, 'html.parser')
side_names=[tag.text.strip() for tag in soup3.select('div.mini-tabcontents div.textbox h4')]
side_prices=[int(tag.text.strip().replace(',','').replace('원 \n\n 해당 가격은 권장 소비자가 입니다.','')) for tag in soup3.select('div.mini-tabcontents div.textbox p.price b')]
# print(chick_prices)
for name, price in zip(side_names, side_prices):
    datas.append({'상품명': name, '가격': price, '종류':'사이드'})

# 소스
for page in [19,20]:
    url = sauce_url.format(page)
    response4 = requests.get(url)
    response4.raise_for_status()
    soup4 = BeautifulSoup(response4.text, 'html.parser')
    sauce_names=[tag.text.strip() for tag in soup4.select('div.mini-tabcontents div.textbox h4')]
    sauce_prices=[int(tag.text.strip().replace(',','').replace('원 \n\n 해당 가격은 권장 소비자가 입니다.','')) for tag in soup4.select('div.mini-tabcontents div.textbox p.price b')]
    for name, price in zip(sauce_names, sauce_prices):
        datas.append({'상품명': name, '가격': price, '종류':'소스'})

    
    # print(df.head())
df = pd.DataFrame(datas, columns=['상품명','가격','종류'])

if __name__ == "__main__":
    print(df.head(3))
    print(f'건수 : {df['상품명'].count()}')
    print(f'가격 평균 : {df['가격'].mean():.2f}')
    print(f'가격 표준편차 : {df['가격'].std():.2f}')
    print(f'가격 최고가격 : {df['가격'].max():.2f}')  
    print(f'가격 최저가격 : {df['가격'].min():.2f}')
