'''
Json : xml에 비해 가벼우며, 배열에 대한 지식만 있으면 처리 가능
'''
import json
'''
dict = {
    'name': 'tom',
    'age': 33,
    'score': ['90', '80', '70'],
}
print('dict:%s' % dict)
# dict:{'name': 'tom', 'age': 33, 'score': ['90', '80', '70']}

print(type(dict)) # <class 'dict'>

# Json encoding : dict를 JSON 문자열로 변경하는 것
str_val = json.dumps(dict) # dict -> str
print(str_val) # dict랑 똑같이 생겼지만 dict랑 엄연히 다름
# {"name": "tom", "age": 33, "score": ["90", "80", "70"]} 

print(type(str_val)) # <class 'str'>
# print(str_val['name']) # TypeError

# Json decoding : JSON 문자열을 dict로 변경하는 것
json_val = json.loads(str_val) # str -> dict
print(json_val)
print(type(json_val)) # <class 'dict'>

for k in json_val.keys(): # keys() : dict 명령어
    print(k)
'''

# 웹에서 Json 문서 읽기
import urllib.request as req
url = 'https://raw.githubusercontent.com/pykwon/python/master/seoullibtime5.json'
plainText = req.urlopen(url).read().decode()
# print(plainText)
print(type(plainText)) # <class 'str'> 으로 할 수 있는게 제한적이다 json 변환

jsonData = json.loads(plainText)
print(type(jsonData)) # <class 'dict'>

print(jsonData['SeoulLibraryTime']['row'][0]['LBRRY_NAME']) # LH강남3단지작은도서관

# dict의 자료를 읽어 도서관명, 전화, 주소 -> DataFrame
libData = jsonData.get('SeoulLibraryTime').get('row')

print(libData[0].get('LBRRY_NAME'))

datas = []
for ele in libData:
    name = ele.get('LBRRY_NAME')
    tel = ele.get('TEL_NO')
    addr = ele.get('ADRES')
    # print(f'이름 :{name}, 전화번호 :{tel}, 주소 :{add}')
    datas.append([name, tel, addr])

import pandas as pd
df = pd.DataFrame(datas, columns=['도서관명','전화','주소'])
print(df)
#                도서관명            전화                                      주소
# 0      LH강남3단지작은도서관   02-459-8700                      서울특별시 강남구 자곡로3길 22
# 1         강남구립못골도서관   02-459-5522                       서울특별시 강남구 자곡로 116
# 2        강남역삼푸른솔도서관  02-2051-1178                 서울특별시 강남구 테헤란로8길 36. 4층
# 3  강남한신휴플러스8단지작은도서관   02-445-9831  서울특별시 강남구 밤고개로27길 20(율현동, 강남한신휴플러스8단지)
# 4    강남한양수자인작은씨앗도서관                                     서울특별시 강남구 자곡로 260





