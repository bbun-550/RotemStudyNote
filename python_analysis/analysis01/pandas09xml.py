'''
XML 문서 처리
'''
import pandas as pd
from bs4 import BeautifulSoup

with open('./my.xml', mode='r', encoding='utf-8') as f:
    xmlfile= f.read()
    # print(xmlfile)

soup = BeautifulSoup(xmlfile, 'xml')  # xml로 읽기
# print(soup.prettify())  # xml 문서 구조 보기
itemTag = soup.findAll('item')  # find_all == findAll
# print(itemTag[0]) # list로 반환
# [<item>
# <name id="ks1">홍길동</name>
# <tel>010-111-1111</tel>
# <exam eng="90" kor="100"/>
# </item>]

nameTag = soup.find_all('name')
# print(nameTag[0].text)  # 홍길동
# print(nameTag[0]['id'])  # ks1

for i in itemTag:
    nameTag = i.find_all('name')
    for j in nameTag:
        # print('id:' + j['id'] + ', 이름:' + j.string)
        print(f'id:{j["id"]}, 이름:{j.text}')  # f-string 사용
        tel = i.find('tel')
        print(f'전화번호:{tel.string}')
    for j in i.find_all('exam'):
        print(f'kor: {j['kor']}, eng: {j['eng']}\n')