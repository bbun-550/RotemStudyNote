# pip install ultralytics opencv-python

# from ultralytics import YOLO
# try:
#     model = YOLO('yolov8n.pt')
# except Exception as e:
#     print(f"error : {e}")

import subprocess
import sys

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    print("ultralytics가 설치되지 않아 설치를 시작합니다.")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics'])
    except subprocess.CalledProcessError as e:
        raise SystemExit("ultralytics 설치 실패! 수동 설치 권장")
    from ultralytics import YOLO

import ultralytics
ultralytics.checks()    # 시스템 환경 점검

try:
    model = YOLO('yolov8n.pt')    # yolo nano version
except Exception as e:
    print(f"error loading model: {e}")
print(model.names)   # COCO dataset(class 80개)
print(len(model.names))    # 80

# 이미지 로딩 후 객체 감지 연습
from PIL import Image    # Pillow 이미지 처리 라이브러리로 파일 열기/변환 담당
import matplotlib.pyplot as plt

image_path = '/Users/bunny/Documents/hyundairotem_aimodel/python_analysis/tensorflow/cat_dog/myimg2.jpg'    # yoloex/dog.jpg
# /Users/bunny/Documents/hyundairotem_aimodel/python_analysis/tensorflow/cat_dog/myimg2.jpg
try:
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    plt.close()
except Exception as e:
    print(f"error : {e}")
    exit()

import cv2    # 컴퓨터비전, 영상처리, 머신러닝 영상관련 기능 제공
import numpy as np    # PIL 이미지를 넘파이 배열로 변환하거나 후속 전처리에 활용

try:
    # 학습된 YOLO 모델로 추론 실행
    results = model(image)
except Exception as e:
    print(f"err during inference : {e}")
    exit()

print(results)    # results.boxes, probs, names, plot, save, show, ....
print(results[0].orig_shape)    # (376, 499)

# Pillow -> numpy 배열로 변환(OpenCV로 처리하기 위함)
image = np.array(image)
print(image.shape)    # (376, 499, 3) : 컬러
print(image[:2,:2])

cropped = image[1800:2400,400:1400]
# 이미지 일부 잘라냄
plt.imshow(cropped)
plt.axis('off')
plt.show()
plt.close()
# 감지된 객체 이미지에 박스 채우기
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# Pillow - RGB(채널순서), OpenCV - BGR(채널순서), matplotlib으로 출력할 때 다시 RGB로

for result in results:
    try:
        for box in result.boxes:    # 바운딩박스 리스트
            # 좌표
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(x1, y1, x2, y2)    # 130 5 298 325
            
            # 객체
            label = result.names[int(box.cls[0])]    # box.cls[0] = tensor([16.])
            print(label)    # result.names의 16번째 value가 dog이다.
            
            # 신뢰도
            confidence = box.conf[0].item()    # float type
            print(f"신뢰도 : {confidence:.2f}")
            
            # Bounding Box 그리기
            cv2.rectangle(image, (x1,y1), (x2,y2), color=(0,255,0), thickness=2)
            cv2.putText(image, f"{label}:{confidence:.2f}", (x1, y1 + 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0,255,0), thickness=5)
    except Exception as e:
        print(f"processing err : {e}")
    
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
plt.close()

# 이미지 저장
cv2.imwrite('yoloex/outtest1.jpg', image)