'''
matplotlib
플로팅 모듈, 다양한 그래프 지원 함수 지원
'''
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic') # 한글 깨짐 방지. mac은 apple gothic
# 한글을 쓰면 음수가 깨짐
plt.rcParams['axes.unicode_minus'] = False # 음수 깨짐 방지

'''
# line plot
x = ['서울', '인천','수원'] # 숫자만 좌표에 올릴 수 있음. 그래서 set 안됨
y = [5,3,7]
plt.xlim([-1,3]) # 경계값 지정. 틱 
plt.ylim([0,10])
plt.plot(x, y)
plt.yticks(list(range(0,10,3))) # 틱 설정
# plt.show()
# jupyter notebook에서는 '%matplotlib inline' 이라고 쓰면 show() 쓸 필요없음
'''

'''
data = np.arange(1,11,2)
# print(data) # [1 3 5 7 9] - 구간 4

plt.plot(data)
x = [0,1,2,3,4]
for a,b in zip(x, data): 
    plt.text(a, b, str(b))
# y축 지정 안하면 알아서 지정해줌

plt.plot(data)
plt.plot(data, data, 'r') # r 선의 색
for a,b in zip(data, data): 
    plt.text(a, b, str(b))
plt.show()
'''

'''
# sin 곡선
x = np.arange(10)
y = np.sin(x)
# print(x, y)
# [0 1 2 3 4 5 6 7 8 9] 
# [ 0. 0.84147098  0.90929743  0.14112001 -0.7568025  -0.95892427 -0.2794155   0.6569866   0.98935825  0.41211849]

# plt.plot(x,y, 'bo') # style 지정 파란색 동그라미 점 표시
# plt.plot(x,y, 'r+') # 빨간색 + 표시
plt.plot(x,y, 'go--', linewidth=2, markersize=12) # 초록 동그라미 점선 표시, 선 두께, 
# - (solid line), -- (dashed line)
# c='b' or color='b'
# linewidth or lw
# marker='o'
# markersize or ms
plt.show()
'''


# 홀드 명령
# 하나의 영역에 두 개 이상의 그래프 표시
x = np.arange(0,3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.figure(figsize=(10,5)) # 그래프 전체 크기 지정
plt.plot(x,y_sin, 'r') # 선
plt.scatter(x,y_cos) # 산점도
plt.xlabel('x축')
plt.ylabel('y축')
plt.title('제목이얌')
plt.legend(['sin','cos']) # 범례
plt.show()


# subplot
# Figure을 여러 개 선언

plt.subplot(2, 1, 1)
plt.plot(x, y_sin)
plt.title('나는 사인이야')
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('나는 코\'사인\'이야')
plt.show()

'''
# 꺾은선 그래프
irum = ['a','b','c','d','e']
kor = [80,50,70,70,90]
eng = [60,70,80,70,60]
plt.plot(irum, kor, 'ro-')
plt.plot(irum, eng, 'gs-') # green squre
plt.ylim([0,100])
plt.legend(['국어','영어'], loc='best') # 범례, 위치(시계 반대방향 순) 2 : 왼쪽 꼭지점
plt.grid(True) # 격자 그리기

fig = plt.gcf() # 차트를 이미지로 저장
plt.show()
fig.savefig('result.png') # plt.show를 감싸면 됨
'''

from matplotlib.pyplot import imread
img = imread('result.png')
plt.imshow(img)
plt.show()