'''
pandas로 파일 저장
'''
import pandas as pd

items = {'apple':{'count':10, 'price':1500},
         'orange':{'count':4, 'price':700}}
df = pd.DataFrame(items)
#        apple  orange
# count     10       4
# price   1500     700

# df.to_clipboard()
# print(df.to_html()) # 앞으로 django로 출력할꺼임
# print(df.to_json()) # AJAX 처리할 때 활용
# {"apple":{"count":10,"price":1500},"orange":{"count":4,"price":700}}

data = df.T
print(data)
# df.to_csv('result1.csv', sep=',', index=False)

'''
엑셀 관련
여러 시트를 가지고 있다
'''
df2 = pd.DataFrame({'name':['Alice','Bob','Oscar'],
                    'age':[24,25,33],
                    'city':['Seoul','Suwon','Incheon']
                    })
#     name  age     city
# 0  Alice   24    Seoul
# 1    Bob   25    Suwon
# 2  Oscar   33  Incheon
# 엑셀에 넣어보자

# 엑셀로 저장
# df2.to_excel('result2.xlsx', index=False, sheet_name='mysheet') 

# 엑셀 읽기
exdf = pd.ExcelFile('result2.xlsx')
print(exdf.sheet_names) # 시트 이름 확인
dfexcel = exdf.parse('mysheet') # 해당 시트 정보 정의
print(dfexcel)
#     name  age     city
# 0  Alice   24    Seoul
# 1    Bob   25    Suwon
# 2  Oscar   33  Incheon
# 3   John   22    Busan # 엑셀에서 수정 저장한 데이터