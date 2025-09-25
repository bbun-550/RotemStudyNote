from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import cifar100

# 데이터 준비
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 정규화
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# label(class) 원핫 처리
NUM_CLASSES = 100
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

model = tf.keras.models.load_model('/Users/bunny/Documents/hyundairotem_aimodel/python_analysis/tensorflow/tf45_ex6_mac.keras')
target_height, target_width = model.input_shape[1:3]

# model.summary()

'''
CLASSES = np.array([
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard',
    'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushrooms', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
    'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
    'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
    'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
    'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train',
    'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf',
    'woman', 'worm'
])


CLASSES = np.array([
    # 수중 포유류
    "beaver", "dolphin", "otter", "seal", "whale",

    # 어류
    "aquarium fish", "flatfish", "ray", "shark", "trout",

    # 꽃
    "orchids", "poppies", "roses", "sunflowers", "tulips",

    # 식품 용기
    "bottles", "bowls", "cans", "cups", "plates",

    # 과일 및 채소
    "apples", "mushrooms", "oranges", "pears", "sweet peppers",

    # 가전 제품
    "clock", "computer keyboard", "lamp", "telephone", "television",

    # 가정용 가구
    "bed", "chair", "couch", "table", "wardrobe",

    # 곤충
    "bee", "beetle", "butterfly", "caterpillar", "cockroach",

    # 대형 육식동물
    "bear", "leopard", "lion", "tiger", "wolf",

    # 대형 인공 야외 구조물
    "bridge", "castle", "house", "road", "skyscraper",

    # 광활한 자연 풍경
    "cloud", "forest", "mountain", "plain", "sea",

    # 대형 잡식·초식동물
    "camel", "cattle", "chimpanzee", "elephant", "kangaroo",

    # 중형 포유류
    "fox", "porcupine", "possum", "raccoon", "skunk",

    # 비곤충 무척추동물
    "crab", "lobster", "snail", "spider", "worm",

    # 사람
    "baby", "boy", "girl", "man", "woman",

    # 파충류
    "crocodile", "dinosaur", "lizard", "snake", "turtle",

    # 소형 포유류
    "hamster", "mouse", "rabbit", "shrew", "squirrel",

    # 나무
    "maple", "oak", "palm", "pine", "willow",

    # 탈것 1
    "bicycle", "bus", "motorcycle", "pickup truck", "train",

    # 탈것 2
    "lawn-mower", "rocket", "streetcar", "tank", "tractor"
])
'''
CLASSES = np.array([
    'beaver', 'dolphin', 'otter', 'seal', 'whale',
    'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
    'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
    'bottles', 'bowls', 'cans', 'cups', 'plates',
    'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
    'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
    'bed', 'chair', 'couch', 'table', 'wardrobe',
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
    'bear', 'leopard', 'lion', 'tiger', 'wolf',
    'bridge', 'castle', 'house', 'road', 'skyscraper',
    'cloud', 'forest', 'mountain', 'plain', 'sea',
    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    'crab', 'lobster', 'snail', 'spider', 'worm',
    'baby', 'boy', 'girl', 'man', 'woman',
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
    'maple', 'oak', 'palm', 'pine', 'willow',
    'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
    'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'
])


pred = model.predict(x_test[:20])
pred_indices = np.argmax(pred, axis=-1)
pred_cla = CLASSES[pred_indices]

actual_indices = np.argmax(y_test[:20], axis=-1)
actual_cla = CLASSES[actual_indices]

print(pred_indices)
print(actual_indices)
print('예측값: ', pred_cla)
print('실제값: ', actual_cla)
print('분류 실패 수: ', np.sum(pred_indices != actual_indices))


IMG_PATH = Path('/Users/bunny/Documents/hyundairotem_aimodel/python_analysis/tensorflow/cat_dog')

image_files = []
if IMG_PATH.exists():
    for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_files.extend(sorted(IMG_PATH.glob(pattern)))

if image_files:
    def load_external_image(path: Path) -> tf.Tensor:
        image_bytes = tf.io.read_file(str(path))
        image = tf.image.decode_image(image_bytes, channels=3)
        image = tf.image.resize(image, (target_height, target_width))
        image = tf.cast(image, tf.float32) / 255.0
        return image

    external_images = tf.stack([load_external_image(path) for path in image_files], axis=0)
    external_pred = model.predict(external_images)
    external_indices = np.argmax(external_pred, axis=-1)
    external_classes = CLASSES[external_indices]

    print('\nExternal image predictions:')
    for path, label, idx in zip(image_files, external_classes, external_indices):
        print(f"{path.name}: {label} (class #{idx})")
else:
    print(f'No image files found in {IMG_PATH}')
