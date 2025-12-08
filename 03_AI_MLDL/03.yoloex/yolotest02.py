'''
# 웹캠을 통해 카메라가 정상적으로 작동 하는지 확인하기
# 웹사이트 (webcamtests.com)에서 카메라 정상 작동 확인 가능
# Python으로 아래 코드를 실행하면 웹캠을 통해 실시간으로 객체를 감지할 수 있다.
import cv2

# macOS에서는 AVFoundation 백엔드를 우선 사용하면 카메라 초기화가 더 안정적이다.
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# 오래된 OpenCV 버전에서는 CAP_AVFOUNDATION이 없을 수 있으므로 기본 방식으로 한 번 더 시도
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("웹캠을 열 수 없습니다. 시스템 환경설정 > 보안 및 개인정보 보호에서 앱의 카메라 접근을 허용했는지 확인하세요.")

print("웹캠이 열렸습니다. ESC를 누르면 종료됩니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽어 올 수 없습니다. 카메라 연결을 확인하세요.")
        break

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 입력 시 종료
        break

cap.release()
cv2.destroyAllWindows()
'''

import cv2
from ultralytics import YOLO
import time
import os
model = YOLO('yolov8n.pt')
print(model.names)

# 감지된 이미지 저장 폴더
save_dir = 'yoloex/test_dir'
os.makedirs(save_dir, exist_ok=True)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
	print('웹캠 사용 불가')
	exit()
else:
	print("웹캠 사용 가능")

cv2.namedWindow("Yolo 실시간 객체 감지", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Yolo 실시간 객체 감지", 800, 600)

# 중복 저장 방지용으로 객체별 마지막 저장 시각 기록
last_saved_time = {}

while True:
    ret, frame = cap.read()    # ret: 프레임 읽기 True/False 반환, frame
    if not ret:
        print("프레임을 읽을 수 없어요")
        break
    
    results = model(frame, verbose=False)    # model.predict(기본값 변경)
    
    # 특정 객체만 감지에 참여
    allowed_labels = [
        'person', 'laptop', 'mouse', 'keyboard', 'cellphone', 'book', 'clock'
    ]
    
    for result in results:
        for box in result.boxes:
            
            # 특정 객체만 감지
            label = result.names[int(box.cls[0])]
            
            # if label != 'person': continue    # person 만 허용
            if label not in allowed_labels: continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])    # bounding box 좌표
            label = result.names[int(box.cls[0])]    # 감지된 객체 이름
            confidence = box.conf[0].item()    # 신뢰도
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}:{confidence:.2f}", (x1, y1 + 10), \
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,255,0), thickness=2)

            # 2초 간격으로 중복 방지 저장
            now = time.time()
            last_time = last_saved_time.get(label, 0)    # 어떤 객체가 처음으로 감지되면 값이 없으니 0을 반환
            
            if now - last_time >= 3:
                filename = f"{label}_{int(now)}.jpg"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, frame)
                print(f"저장 성공! : {filepath}")
                last_saved_time[label] = now
                
    # 감지된 프레임 화면에 출력
    cv2.imshow('YOLO 객체 실시간 감지', frame)
    
    key = cv2.waitKey(1)    # 1ms  동안 입력 대기, 아무키도 안 누르면 -1 반환
    
    if key != -1:
        print(f'눌린 키: {key}, {chr(key)}')
        
    print(f'눌린 키: {key}')
    
    if key & 0xFF == ord('q'):    # 화면이 뜨고 나서 화면 안에서 q가 눌리면 종료
        # ord('a') -> 97 반환
        break

# 자원 정리
cap.release()    # 사용 중인 카메라 장치(점유) 해제
cv2.destroyAllWindows()
