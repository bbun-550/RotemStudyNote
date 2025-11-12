import os, json, random, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v2

IMG_SIZE = (224, 224)

def load_class_names(path='class_name.txt'):
    with open(path, mode='r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    return names

# 모델이 기대하는 입력 차원과 형식을 맞춰주는 전처리 단계 함수
def load_and_preprocess(img_path):
    img = keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = keras.utils.img_to_array(img)    # 이미지를 float32 배열(0~255)로 변환
    arr = np.expand_dims(arr, axis=0)    # 배열 차원 추가  (244, 244, 3) -> (1, 244, 244, 3)
    return arr

def main():
    if len(sys.argv) < 2:
        print('분류할 파일명.확장자 입력하세요')
        sys.exit(1)
        
    image_path = sys.argv[1]
    # print(image_path)
    
    # 이미지 분류 모델 로딩
    model = keras.models.load_model('/Users/bunny/Documents/hyundairotem_aimodel/python_analysis/best_model.keras', 
                                         compile=False, custom_objects = {'preprocess_input':mobilenet_v2.preprocess_input})
    
    
    # class_name.txt를 읽어 인덱스 -> 클래스명과 매핑
    class_names = load_class_names('/Users/bunny/Documents/hyundairotem_aimodel/python_analysis/class_name.txt')
    # print(class_names) # ['etc', 'glass', 'metal', 'paper', 'plastic']
    x = load_and_preprocess(image_path)
    
    preds = model.predict(x)[0]
    print(preds)
    
    top_idx = int(np.argmax(preds))
    top_prob = float(preds[top_idx])
    
    # 상위 3개 클래스 출력
    print(f"예측값 : {class_names[top_idx]}({top_prob*100:.3f}%)")
    order = np.argsort(-preds)      # - 오름차순을 내림차순으로 변경해줌
    print("분류 예측 결과 상위 3개 :")
    for n,i in enumerate(order[:3]):
        print(f"{n+1}. {class_names[i]}({preds[i]*100:.3f}%)")



if __name__ == '__main__':
    main()
    
# 실행(터미널)에서 > python classify.py 파일명.jpg
# - python classify.py data/train/metal/metal1.jpg    

