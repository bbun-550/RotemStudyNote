# 손글씨(숫자 이미지) 읽기
from PIL import Image # Python Imaging Library 파이썬에서 이미지 처리를 위한 라이브러리 : 이미지 파일 로드, 편집하거나 필터 적용으로 다양한 이미지 처리 작업 기능 수행
import numpy as np
import matplotlib.pyplot as plt

im = Image.open('/Users/bunny/Documents/git_practice/python_analysis/tensorflow/digits/6.png')

# 원래 이미지 크기를 28 * 28 크기로 리사이즈 (이유 : MNIST 기준)
# 흑백(0 ~ 255)으로 변환 후 numpy배열로 변환

img = np.array(im.resize((28,28), Image.Resampling.LANCZOS).convert('L'))
print(img.shape)

plt.imshow(img, cmap='Greys')
plt.show()
plt.close()

# (28*28) 이미지를 (1, 784) 벡터로 변환 (Dense 클래스 입력 형태)
data = img.reshape([1,784]).astype('float32')
# print(data)

data = data / 255.0 # 픽셀 값을 0 ~ 1 범위로 정규화
print(data)

# 다시 시각화 (1, 784) => (28,28) reshape
plt.imshow(data.reshape(28,28), cmap='Greys')
plt.show()