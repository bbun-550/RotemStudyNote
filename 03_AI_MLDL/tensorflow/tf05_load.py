import numpy as np
from keras.models import load_model

x = np.array([[0,0],[0,1],[1,0],[1,1]])

# 8 모델 읽기
model = load_model('test.keras')

proba = model.predict(x, verbose=0)
print(proba)

pred = (proba > 0.5).astype('int32')
print(pred.ravel())
