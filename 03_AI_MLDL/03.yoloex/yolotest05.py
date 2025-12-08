# 이미지 detection + TTS
# 유기동물 사진을 제출하면 탐지 후 안내소로 안내하는 문구를 소리로 표현
# pip install playsound==1.2.2 gTTS==2.5.4 pyobjc

from gtts import gTTS
# from IPython.display import Audio     # Jupyter notebook
from playsound import playsound    # local
'''
# 작동 테스트

def speak_shelter_info(message):
    tts = gTTS(text=message, lang='ko')
    tts.save('yoloex/yolotest5_sound.mp3')
    playsound('yoloex/yolotest5_sound.mp3')
    os.system("afplay yoloex/yolotest5_sound.mp3")   # 맥OS 기본 mp3 플레이어 호출
    

message = '30일 한경닷컴이 채용 플랫폼 진학사 캐치에 의뢰한 SNS 사용실태 설문 결과 Z세대 10명 중 7명꼴로 이번 카카오톡 업데이트를 부정적으로 평가했다'
speak_shelter_info(message=message)
'''

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime

def show_shelter_info_func(region, shelters, detected_info):
    shelter_info = shelters.get(region, shelters['기본'])
    pet_summary = f"{detected_info['count']}마리 ({', '.join(detected_info['labels'])})"
    message = (
        f"유기동물 탐지 결과:\n"
        f"- 탐지된 동물 수 : {detected_info['count']}\n"
        f"- 종류 : {', '.join(detected_info['labels'])}\n"
        f"{region} 지역 보호소 정보 : {shelter_info}"
    )
    print('보호소 정보 :')
    print(message)    # 문자 안내
    
    # 음성 안내
    try:
        tts = gTTS(text=f"{region} 지역에 유기된 {pet_summary}가 감지되었습니다. 가까운 보호소는 {shelter_info} 입니다", lang='ko')
        tts.save('yoloex/yolotest5_sound.mp3')    # 음성 파일 저장
        playsound('yoloex/yolotest5_sound.mp3')    # 음성 파일 재생
    except Exception as e:
        print(f"음성안내 실패 : {type(e).__name__} - {e}")
    

def handle_stray_pet_func(region, shelters, detected_info):
    print("유기 동물로 추정됩니다")
    show_shelter_info_func(region, shelters, detected_info)
    

'''
# YOLO로 탐지하기 전 음성 테스트

region = "강남"
shelters = {    # 보호소 정보
    "서울":"서울 반려동물 보호센터 : 02-1234-1234",
    "기본":"전국 반려동물 보호센터 : 1577-8214",
}

# YOLO로 알 수 있다.
detected_info = {
    "count":3,
    "labels":['호랑이','사자','코끼리']
}

handle_stray_pet_func(region, shelters, detected_info)
'''

# 탐지 정보 로그 파일로 저장
def save_detection_log_func(image_path, detection_data):
    log_file_name = 'yoloex/yolotest5_detec.txt'
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(log_file_name, 'a', encoding='utf-8') as f:
        f.write(f"[{now}] 이미지: {image_path}\n")
        f.write(f"탐지된 객체 수: {len(detection_data)}\n")
        
        for d in detection_data:
            f.write(f" - {d['label']}: box={d['box']}, confidence:{d['confidence']:.2f}\n")
        
        f.write("-"*40 + "\n")
    print(f"탐지 결과가 {log_file_name}에 저장되었습니다.")


# 유기동물 감지 함수
def detect_pets_func(image_path):
    pet_desc = {
        'dog':'강아지',
        'cat':'고양이',
        'cow':'소',
        'horse':'자연산 1馬력 엔진',
        'elephant':'코끼리',
        'giraffe':'기린',
        'zebra':'얼룩말',
        'bird':'새',
        'bear':'곰은 사람을 찢어',
        'sheep':'맛있는 양꼬치',
    }
    
    shelters = {    # 보호소 정보
    "서울":"서울 반려동물 보호센터 : 02-1234-1234",
    "부산":"부산 유기동물 보호소 : 051-4321-4321",
    "기본":"전국 반려동물 보호센터 : 1577-8214",
    }
    
    stray_keywords = ['street', 'road', 'outside', 'stray']
    
    model = YOLO('yolov8n.pt')
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 부를 수 없습니다.")
        return

    results = model(image)
    detected_pets = []
    detection_data = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int,box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0].item()
            
            if label in pet_desc:
                detected_pets.append(label)
                detection_data.append(
                    {
                        'label': pet_desc.get(label, label),
                        'box': (x1, y1, x2, y2),
                        'confidence': confidence,
                    }
                )
                cv2.rectangle(image, (x1,y1), (x2,y2), color=(0,255,0), thickness=2)
                # 글자 크기 구하기
                (text_w, text_h), baseline = cv2.getTextSize(f"{label}:{confidence:.2f}", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2)
                cv2.rectangle(image,(x1, y1 - text_h - baseline),(x1 + text_w, y1),(0, 255, 0),-1)
                cv2.putText(image,f"{label}:{confidence:.2f}",(x1, y1 - baseline),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.8,color=(255, 255, 255),thickness=2,lineType=cv2.LINE_AA)
    
    # # 결과 이미지 저장
    # output_path = 'yoloex/test_dir'
    # cv2.imwrite(output_path, image)
    # print(f"감지된 이미지 파일로 저장 성공! {output_path}")
    
    # 이미지 보기
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    plt.close()
    
    if detected_pets:    # detected_pets 리스트에 감지된 동물이 있으면 실행
        print("감지된 동물 결과: ")
        for pet in set(detected_pets):
            print(f"- {pet_desc.get(pet, pet)}")
            
        # 감지된 정보들을 텍스트 파일로 저장
        save_detection_log_func(image_path, detection_data)
        
        # 유기동물 조건 확인
        # 이미지 경로에 stray_keyword에 등록된 단어가 있는 경우
        if any(pet in ['elephant','giraffe'] for pet in detected_pets) and any(keyword in image_path.lower() for keyword in stray_keywords):
            detected_info = {
                'count' : len(detection_data),
                'labels' : sorted(set([d['label'] for d in detection_data]))
            }
            handle_stray_pet_func(region='서울', shelters=shelters, detected_info=detected_info)
        else:
            print("유기동물이 감지되지 않았습니다.")
        
    
image_storage = '/Users/bunny/Documents/hyundairotem_aimodel/python_analysis/yoloex/image_storage/'
test_image = 'stray_pet_detection2.jpg'
detect_pets_func(image_storage + test_image)
