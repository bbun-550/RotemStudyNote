import os, json, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print(BASE_DIR)    # /Users/bunny/Documents/hyundairotem_aimodel/python_analysis/tf02
DATA_DIR_TRAIN = os.path.join(BASE_DIR, 'data', 'train')    # /Users/bunny/Documents/hyundairotem_aimodel/python_analysis/tf02/data/train
# print(DATA_DIR_TRAIN)
DATA_DIR_VAL = os.path.join(BASE_DIR, 'data', 'validation') 


IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 30
LR = 1e-3

# Training Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_TRAIN,
    image_size = IMG_SIZE,
    batch_size = BATCH,
    shuffle = True,
    seed = SEED
)

# Validation Dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_VAL,
    image_size = IMG_SIZE,
    batch_size = BATCH,
    shuffle = False,    # 검증 데이터는 섞지 않는다
    seed = SEED
)

classes_names = train_ds.class_names
num_classes = len(classes_names)
# print(f"Classes : {classes_names}, {num_classes}")
# ['etc', 'glass', 'metal', 'paper', 'plastic'], 5

AUTOTUNE = tf.data.AUTOTUNE     # 대문자 ; final static, 절대 바꾸지 않을 값, 상수

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# 데이터 변형 또는 증강
data_augumentation = keras.Sequential([
    layers.RandomFlip('horizontal'),    # 좌우반전
    layers.RandomRotation(0.05),    # 소량 회전
    layers.RandomZoom(0.1),    # 소량 줌
])

preprocess = mobilenet_v2.preprocess_input    # [-1, 1] 범위 스케일링

# 모델 구성
base = mobilenet_v2.MobileNetV2(
    include_top = False,
    weights = 'imagenet',
    input_shape = IMG_SIZE + (3,)
)

base.trainable = False

# 나의 모델 생성
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augumentation(inputs)     # 입력에 증강을 적용
x = layers.Lambda(preprocess)(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(5, activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=3, restore_best_weights=True)
]

history = model.fit(train_ds, validation_data=val_ds, epochs=3, batch_size=BATCH, verbose=2, callbacks=callbacks)

# 평가
loss, acc = model.evaluate(val_ds)
print(f"acc : {acc:.4f}")
print(f"loss : {loss:.4f}")
# acc : 0.9553
# loss : 0.1350

# 미세조정
# model_load = keras.models.load_model('best_model.keras')

unfreeze_from = 100

for layer in base.layers[unfreeze_from:]:
    layer.trainable = True
    
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

fine_history = model.fit(train_ds, validation_data=val_ds, epochs=3, batch_size=BATCH, verbose=2, callbacks=callbacks)

# 미세조정 평가
loss, acc = model.evaluate(val_ds)
print(f"Final acc : {acc:.4f}")
print(f"Final loss : {loss:.4f}")

with open('class_name.txt', mode='w', encoding='utf-8') as f:
    for name in classes_names:
        f.write(f"{name}\n")