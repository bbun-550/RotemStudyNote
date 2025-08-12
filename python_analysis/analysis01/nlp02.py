'''
pip install pygame
pip install pytagcloud
pip install simplejson
'''
# 동아일보 검색 기능으로 문자열을 읽어 형태소 분석 후 워드 클라우드 출력
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote # 인코딩

# keyword=input('검색어:')
keyword = '삼성전자'

target_url = 'https://www.donga.com/news/search?query=' + quote(keyword) # 이대로 넘어가면 검색 안됨. 인코딩해야함
# print(target_url)
# 삼성전자 -> %EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90

source_code = urllib.request.urlopen(target_url)

soap = BeautifulSoup(source_code, 'lxml', from_encoding='utf8')
# print(soap)