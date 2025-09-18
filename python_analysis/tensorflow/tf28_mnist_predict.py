# mnist로 학습된 모델로 내가 그린 숫자 이미지 분류 확인

from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

loadModel = load_model('/Users/bunny/Documents/git_practice/python_analysis/tensorflow/tf27model.keras')

im = Image.open('/Users/bunny/Documents/git_practice/python_analysis/tensorflow/digits/.png')
img = np.array(im.resize((28,28), Image.Resampling.LANCZOS).convert('L'))
# convert('L') : 이미지를 8비트 그레이스케일(Luminance)로 바꿔서 한 채널(밝기 정보)만 남기기 위한 처리
# Image.Resampling.LANCZOS는 이미지 크기를 바꿀 때 쓰는 리샘플링 필터로, 
#   Lanczos 알고리즘(고차 sinc 기반)을 이용해 고해상도 → 저해상도 변환 시 계단 현상이나 aliasing을 줄이고, 
#   비교적 선명하고 자연스러운 결과를 제공
print(img.shape)

data = img.reshape([1,784]).astype('float32')

pred = loadModel.predict(data)

print(f"예측값 : {np.argmax(pred, 1)}")
plt.imshow(data.reshape(28,28), cmap='Greys')
plt.show()
