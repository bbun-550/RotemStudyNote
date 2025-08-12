'''
pip install konlpy
'''

from konlpy.tag import Okt, Kkma, Komoran

# corpus(말뭉치) : 자연어 처리를 목적으로 수집된 문자 집합
text = '나는 오늘 아침에 강남에 갔다. 가는 길에 파리바게트가 보여 손이 떨렸다. 도벽이 도졌다.'

# 형태소 : 형태소(morpheme)는 언어학에서 (일반적인 정의를 따르면) 일정한 의미가 있는 가장 작은 말의 단위로 발화체 내에서 따로 떼어낼 수 있는 것을 말한다. 
# 즉, 더 분석하면 뜻이 없어지는 말의 단위이다
# Okt
'''
okt = Okt() # Kkma가 빠르지만 Okt를 가장 많이 씀
print(f'형태소 : {okt.morphs(text)}\n')
# 형태소 : ['나', '는', '오늘', '아침', '에', '강남', '에', '갔다', '.', '가는', '길', 
# '에', '파리바게트', '가', '보여', '손', '이', '떨렸다', '.', '도벽', '이', '도', '졌다', '.']

print(f'품사 태깅 : {okt.pos(text)}\n')

print(f'품사 태깅 (어간 포함) : {okt.morphs(text, stem=True)}\n') # stem : 서술어 원형/어근으로 출력해라

print(f'명사 : {okt.nouns(text)}\n') # 명사 출력
# 명사 : ['나', '오늘', '아침', '강남', '길', '파리바게트', '손', '도벽', '도']

kkma = Kkma()
print(f'꼬꼬마 형태소 : {kkma.morphs(text)}\n')

print(f'꼬꼬마 품사 태깅 : {kkma.pos(text)}\n')

print(f'꼬꼬마 명사 : {kkma.nouns(text)}\n') # 명사 출력


komoran = Komoran()
print(f'코모란 형태소 : {komoran.morphs(text)}\n')

print(f'코모란 품사 태깅 : {komoran.pos(text)}\n')

print(f'코모란 명사 : {komoran.nouns(text)}\n') # 명사 출력
'''

# 워드 클라우드
# pip install wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt # 워드 클라우드 시각화
text2 = '나는 오늘 아침에 강남에 갔다. 강남에 있는 강남 빵집에 바게트 빵이 맛스럽게 생겼다. 빵집 이름이 파리바게트 강남점이었다.'

okt = Okt()
nouns = okt.nouns(text2)
words = ' '.join(nouns)
print(f'words : {words}\n')

wc = WordCloud(font_path='malgun.ttf', height=300, background_color='white') # 글꼴 : 맑은 고딕; 높이
cloud = wc.generate(words)

plt.imshow(cloud, interpolation='bilinear') # 바짝바짝 붙이라는 기본 옵션; 이미지를 부드럽게 출력하게 해줌
plt.axis('off') # x,y 좌표는 꺼버려
plt.show()
