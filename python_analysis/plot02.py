'''

'''
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(10)
'''
# figure 구성 방법
# 1. matplotlib 스타일 인터페이스
plt.figure()
plt.subplot(2,1,1) # row, column, panel number
plt.plot(x, np.sin(x))

plt.subplot(2,1,2) # row, column, panel number
plt.plot(x, np.cos(x))
plt.show()

# 2. 객체 지향 인터페이스
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
plt.show()
'''

fig = plt.figure() # 명시적으로 영역 객체 선언
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.hist(np.random.randn(10), bins=5, alpha=0.5) # alpha 투명도, hist 히스토그램
ax2.plot(np.random.randn(10))
plt.show()

# 막대그래프 bar
data = [50, 80, 100, 70, 90]
plt.bar(range(len(data)), data) # 세로 막대그래프
plt.show()

loss = np.random.rand(len(data))
plt.barh(range(len(data)), data, xerr=loss, alpha=0.3) # 가로 막대그래프
plt.show()

# 파이 그래프
plt.pie(data, explode=(0,0.1,0,0,0), colors=['yellow','blue','magenta','green','pink']) # explode : 조각을 분리할꺼야
plt.show()

# boxplot
# 사분위 등에 의한 데이터 분포 확인에 효고적
plt.boxplot(data)
plt.show()

# bubble 차트
# 데이터 크기에 따라서 버블의 크기도 달라짐
n = 30
np.random.seed(47)
x = np.random.rand(n)
y = np.random.rand(n)
color = np.random.rand(n)
scale = np.pi * (15 * np.random.rand(n)) ** 2
plt.scatter(x,y, c=color, s=scale)
plt.show()

# 시계열 데이터
import pandas as pd

fdata = pd.DataFrame(np.random.randn(1000,4), # 1000행 4열
                     index = pd.date_range('1/1/2000', periods=1000), columns=list('ABCD'))
fdata = fdata.cumsum() # 누적합 만든다
print(fdata.head())
plt.plot(fdata)
plt.show()

# pandas가 지원하는 plot
fdata.plot()
fdata.plot(kind='box') # kind : 그래프의 종류 선택
fdata.plot(kind='bar')
plt.xlabel('time')
plt.ylabel('data')
plt.show()
