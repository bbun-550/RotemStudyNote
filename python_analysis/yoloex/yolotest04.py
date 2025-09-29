# 탐지된 객체에 대한 설명을 글로 달아주기
import cv2
from ultralytics import YOLO
import urllib
import matplotlib.pyplot as plt
import datetime


# 객체 설명과 링크 제공
object_info = {
    "person": {
        "description": "이 객체는 사람이 감지된 경우입니다. 사람 감지는 보안 감시, 출입 관리 시스템 등에 매우 유용합니다. 또한 얼굴 인식, 행동 분석 등 다양한 분야에 적용됩니다.",
        "use_case": "사람 감지는 보안 시스템에서 출입 관리, 비상 상황에서의 대처, 헬스케어 분야에서 노인 및 환자의 상태 모니터링에 사용됩니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("사람"))
    },
    "car": {
        "description": "이 객체는 자동차가 감지된 경우입니다. 자동차 감지는 교통 흐름 분석, 불법 주차 감시, 사고 예방 등 다양한 분야에 활용됩니다.",
        "use_case": "자동차 감지는 자율 주행 시스템, 스마트 교통 시스템, 교차로 모니터링 등에 활용되며, 도시 계획 및 교통 관리에도 중요한 역할을 합니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("자동차"))
    },
    "truck": {
        "description": "이 객체는 트럭이 감지된 경우입니다. 트럭 감지는 물류 창고 관리, 도로 교통 모니터링, 고속도로에서의 추적 등에 활용됩니다.",
        "use_case": "트럭 감지는 물류 효율화, 고속도로 사고 예방, 교통량 분석 등에 사용되며, 스마트 물류 및 재난 관리 시스템에도 중요합니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("트럭"))
    },
    "motorcycle": {
        "description": "이 객체는 오토바이가 감지된 경우입니다. 오토바이 감지는 교통 사고 예방 시스템, 도로에서의 차량 추적 등에 사용됩니다.",
        "use_case": "오토바이 감지는 도로 교통 사고 예방, 긴급 상황 대응, 스마트 교통 시스템 등에 사용됩니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("오토바이"))
    },
    "dog": {
        "description": "이 객체는 강아지가 감지된 경우입니다. 강아지 감지는 반려동물 보호, 유기 동물 탐지 및 동물원 관리 등에서 중요합니다.",
        "use_case": "강아지 감지는 동물 보호 시스템, 유기 동물 탐지 시스템 및 스마트 펫 모니터링 시스템에 사용됩니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("강아지"))
    },
    "cat": {
        "description": "이 객체는 고양이가 감지된 경우입니다. 고양이 감지는 스마트 펫 모니터링 시스템과 연계되어 유용하게 사용됩니다.",
        "use_case": "고양이 감지는 반려동물 모니터링 시스템, 동물원 관리 및 스마트 홈 시스템에 활용됩니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("고양이"))
    },
    "bus": {
        "description": "이 객체는 버스가 감지된 경우입니다. 버스 감지는 대중교통 분석, 버스 전용차로 감시 및 혼잡도 모니터링 등에 활용됩니다.",
        "use_case": "버스 감지는 스마트 시티 교통 시스템, 버스 정류장 혼잡도 분석 및 통근 시간 최적화에 사용됩니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("버스"))
    },
    "bird": {
        "description": "이 객체는 새가 감지된 경우입니다. 새 감지는 자연 생태 모니터링, 조류 충돌 방지 시스템 등에 활용됩니다.",
        "use_case": "새 감지는 공항의 조류 충돌 방지 시스템, 야생 동물 보호 구역의 생태계 분석, 스마트 환경 감시 시스템에 활용됩니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("새"))
    }
}

model = YOLO('yolov8n.pt')

image_path = 'yoloex/image4.jpg'    # yoloex/image3.jpeg
image = cv2.imread(image_path)

if image is None:
    print("이미지 읽기 실패!")
    exit()

results = model(image)

detected_obj = []
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0].item()
        detected_obj.append(label)
        
                # 바운딩 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 레이블 텍스트 내용
        text = f"{label}:{confidence:.2f}"

        # 글자 크기 구하기
        (text_w, text_h), baseline = cv2.getTextSize(
            text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, thickness=1
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
            fontScale=0.3,
            color=(255, 255, 255),  # 흰색 글자
            thickness=1,
            lineType=cv2.LINE_AA
        )
# print(f"detected_obj : {detected_obj}")
'''
0: 352x640 4 persons, 3 cars, 2 motorcycles, 48.4ms
Speed: 3.3ms preprocess, 48.4ms inference, 10.2ms postprocess per image at shape (1, 3, 352, 640)
detected_obj : ['person', 'person', 'motorcycle', 'person', 'motorcycle', 'person', 'car', 'car', 'car']
'''

# 결과 이미지 저장 (시간 별 저장)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
# print(f"timestamp : {timestamp}")

output_path = f"yoloex/yolotest04_{timestamp}.jpg"
cv2.imwrite(output_path, image)
print(f"탐지된 객체가 {output_path}로 저장됨")

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
plt.close()

# 감지된 이미지에 설명 및 링크 출력
description_text = ""

for obj in set(detected_obj):
    if obj in object_info:
        description_text += f"\n{obj} 탐지됨:\n"
        description_text += f"설명:{object_info[obj]['description']}\n"
        description_text += f"사용사례:{object_info[obj]['use_case']}\n"
        description_text += f"자세한 내용:{object_info[obj]['link']}\n"
        
print(f"\n객체 설명 : {description_text}")

# 감지 결과 로그 파일로 저장
# - db에 저장할 수도 있다
log_file = "yoloex/yolotest04log.txt"
with open(log_file, 'a', encoding='utf-8') as log:
    log.write(f"[{timestamp}]감지된 객체 : {', '.join(set(detected_obj))}\n")
    log.write(description_text + '\n\n')

print(f"{log_file}에 저장!")