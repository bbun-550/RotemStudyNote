import os
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

model = YOLO('yolov8n.pt')

image_path = 'yoloex/people.webp' # myimg2.jpg

try:
    image = cv2.imread(image_path)
    
except Exception as e:
    print(f"에러 : {e}")
    raise SystemExit

original = image.copy() # 원본 이미지 저장

results = model(image)
print(results)

# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
# plt.close()

# 감지된 사람 수

person_count = 0
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0].item()
        
        if label.lower() == 'person':
            person_count += 1
        
        # 바운딩 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 레이블 텍스트 내용
        text = f"{label}:{confidence:.2f}"

        # 글자 크기 구하기
        (text_w, text_h), baseline = cv2.getTextSize(
            text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2
        )

        # 레이블 박스 좌표 계산 (x1,y1 기준 위쪽)
        cv2.rectangle(
            image,
            (x1, y1 - text_h - baseline),   # 왼쪽 위
            (x1 + text_w, y1),              # 오른쪽 아래
            (0, 255, 0),                    # 초록색
            -1                              # 채우기
        )

        # 글자 그리기 (박스 위에 흰색으로)
        cv2.putText(
            image,
            text,
            (x1, y1 - baseline),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(255, 255, 255),  # 흰색 글자
            thickness=2,
            lineType=cv2.LINE_AA
        )


print(f"감지된 사람 수: {person_count}명")

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f"detected person ({person_count})")
plt.show()
plt.close()

# 바운딩 박스된 이미지 전체를 저장
out_path = 'yoloex/test_dir/yotest3_out.jpg'
cv2.imwrite(out_path, image)
print('저장완료!')

# 바운딩 박스 내부 객체만 저장(부분 저장)
for idx, result in enumerate(results):
    for j, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0].item()
        
        # 원본 이미지에서 ROI(Region of Interest) 추출
        cropped = image[y1:y2, x1:x2]    # image(H, W, 3) 배열 슬라이싱을 통해 선택된 이미지 배열 반환
        # print(f"cropped : {cropped}")
        
        # 선택된 이미지 배열 저장
        crop_path = f"yoloex/test_dir/crop_{idx}_{j}_{label}_{confidence:.2f}.jpg"
        # os.path.join()를 사용해도 된다.
        
        cv2.imwrite(crop_path, cropped)
        print(f"객체 {label}이 저장 성공!")
  
  
# 바운딩 박스 내부 객체만 저장(박스 선 없이 저장)
# os.makedirs('crops')
      
for idx, result in enumerate(results):
    for j, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0].item()
        
        # 원본 이미지에서 ROI(Region of Interest) 추출
        cropped = original[y1:y2, x1:x2]    # 초록선 없는 이미지(원본 이미지를 좌표로 추출)
        # print(f"cropped : {cropped}")
        
        # 선택된 이미지 배열 저장
        crop_path = f"yoloex/test_dir/crop_ori_{idx}_{j}_{label}_{confidence:.2f}.jpg"
        # os.path.join('yoloex/test_dir', f"crop_ori_{idx}_{j}_{label}_{confidence:.2f}.jpg")를 사용해도 된다.
        
        cv2.imwrite(crop_path, cropped)
        print(f"객체 {label}이 저장 성공!")
        
        
# 감지된 객체의 중심 좌표 출력
p_count = 0
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0].item()
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        if label.lower() == 'person':
            p_count = p_count + 1
            print(f"Person => {p_count} : 중심좌표는 ({center_x}, {center_y}), 신뢰도:{confidence:.2f}")
            
            # 중심점 그리기
            cv2.circle(image, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)
            
            coord_text = f"({center_x}, {center_y})"
            cv2.putText(image, coord_text, (center_x + 10, center_y), \
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
           
        
         
        # 바운딩 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}:{confidence:.2f}"

        # 글자 크기 구하기
        (text_w, text_h), baseline = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2)

        # 레이블 박스 좌표 계산 (x1,y1 기준 위쪽)
        cv2.rectangle(image,(x1, y1 - text_h - baseline),   # 왼쪽 위
            (x1 + text_w, y1),              # 오른쪽 아래
            (0, 255, 0),                    # 초록색
            -1                              # 채우기
        )

        # 글자 그리기 (박스 위에 흰색으로)
        cv2.putText(image,text,(x1, y1 - baseline),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(255, 255, 255),  # 흰색 글자
            thickness=2,
            lineType=cv2.LINE_AA
        )

plt.figure(figsize=(10,8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
plt.close()