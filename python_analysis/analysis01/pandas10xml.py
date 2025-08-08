'''
XML로 제공되는 날씨자료
'''
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import pandas as pd

# url_gangnam = 'http://www.kma.go.kr/wid/queryDFSRSS.jsp?zone=1159064000'  # 서울시 강남구 역삼동
# data = urllib.request.urlopen(url_gangnam).read()
# print(data.decode('utf-8'))  # 깨져보일 수 있어서 utf-8로 디코딩

url = 'http://www.kma.go.kr/XML/weather/sfc_web_map.xml' # 기상청에서 제공하는 XML 날씨자료    
# data = urllib.request.urlopen(url).read()
# print(data.decode('utf-8'))  # 깨져보일 수 있어서 utf-8로 디코딩

soup = BeautifulSoup(urllib.request.urlopen(url), 'xml')  # xml로 읽기
# print(soup.prettify())  # xml 문서 구조 보기


# 지역 , 온도 , desc dataframe 생성
'''
loc = soup.findAll('local')
# print(loc[:3])
data = []
for i in loc:
    name = i.text.strip()
    temp = i.get('ta')
    desc = i.get('desc')
    data.append((name, temp, desc))

df = pd.DataFrame(data, columns=['지역', '온도', '날씨'])
# print(df.head())
df.to_csv('weather.csv', index=False, encoding='utf-8-sig')  # CSV 파일로 저장
'''

df = pd.read_csv('weather.csv', encoding='utf-8')  # CSV 파일로 읽기
print(df.head(3))
print(df[0:3])
print(df.iloc[0:3,:])
print(df.loc[0:3, ['지역', '온도','날씨']])

#     지역    온도    날씨
# 0   속초  26.6  구름많음
# 1  북춘천  28.3  구름조금
# 2   철원  28.6  구름조금

print(df.tail(2))
print(df[-2:len(df)])
#      지역    온도  날씨
# 95   남해  31.0  흐림
# 96  북부산  32.1  흐림

df.info()
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   지역      97 non-null     object
#  1   온도      97 non-null     float64
#  2   날씨      97 non-null     object
# dtypes: float64(1), object(2)

print(f'온도 평균 : {df['온도'].mean():.2f}도')
# 온도 평균 : 29.25도

print(df['온도'] >= 33)
print(df.loc[df['온도'] >= 33, ['지역', '온도','날씨']]) # 온도 33도 이상인 지역,온도,날씨 출력
#     지역    온도    날씨
# 92  밀양  33.1  구름많음

print(df.sort_values(['온도'], ascending=True)) # 온도 오름차순 정렬
#      지역    온도    날씨
# 5   대관령  24.2  구름많음
# 7   백령도  25.6  구름조금
# 10   동해  26.3    흐림
# 0    속초  26.6  구름많음