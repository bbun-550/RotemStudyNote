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

# headers = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request(target_url) # , headers=headers
source_code = urllib.request.urlopen(req)

soup = BeautifulSoup(source_code, 'lxml', from_encoding='utf8')
# print(soup.prettify())

msg = ""
for title in soup.findAll('h4', class_='tit'):
    title_link = title.select('a')
    # print(title_link)
# [<a data-ep_button_area="검색결과" ... “TV 광원소자 더 작게  더 밝게” 기술로 中 따돌린다</a>]

    article_url = title_link[0]['href']
    # print(article_url) # https://www.donga.com/news/Economy/article/all/20250812/132175436/2

    try:
        source_article = urllib.request.urlopen(article_url)
        soup = BeautifulSoup(source_article, 'lxml', from_encoding='utf8')
        contents = soup.select('section.news_view')
        # print(contents) # 안에 텍스트만 불러오기. 중간중간 광고 있음

        for imsi in contents:
            item = str(imsi.find_all(string=True))
            # print(item) # 이스케이프 없애는 법(정규표현식). 하지만 형태소 분석으로 자를꺼임
            msg += item

    except Exception as e:
        pass

# print(msg) # 가져왔으니까 이제 가공

from konlpy.tag import Okt
from collections import Counter # 단어 개수 세는 용도

okt = Okt() # okt 인스턴스 했음
nouns = okt.nouns(msg) # 명사만 잡아올 수 있다

result = []
for imsi in nouns:
    if len(imsi) > 1: # 한 글자짜리는 쳐냄. 두 글자 이상만 취급
        result.append(imsi)

# print(result[:5]) # ['삼성', '마이크로', '출시', '화질', '직결']

# 너무 많이 나오면 안되니까 
count = Counter(result)
# print(count) # Counter({'인재': 34, '전자': 32, '측정': 31, ...

tag = count.most_common(50) # 상위 50개만 출력

# wordcloud 만들기
import pytagcloud
taglist = pytagcloud.make_tags(tag, maxsize=100)
# print(taglist) # [{'color': (94, 199, 88), 'size': 127, 'tag': '인재'}, ...

pytagcloud.create_tag_image(taglist, 'word.png', size=(1000,600), 
                            background=(0,0,0), rectangular=False,
                            fontname='Malgun') # 이미지로 저장. 꼭 fontname= 지정해줘야함

# 이미지 읽기
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('word.png')
plt.imshow(img)
plt.show()