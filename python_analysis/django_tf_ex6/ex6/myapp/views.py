import io  # 플롯 이미지를 메모리 버퍼에서 직렬화할 때 사용하는 바이트 스트림 제공
import uuid  # 저장 파일에 고유 ID를 부여하기 위한 유틸리티
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 서버 환경에서 사용 가능한 비대화형 백엔드를 강제로 지정
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from django.core.files.base import ContentFile  # 원시 바이트 데이터를 Django 스토리지용 객체로 감싸기
from django.core.files.storage import FileSystemStorage  # MEDIA_ROOT에 파일을 저장하고 관리
from django.shortcuts import render


model = tf.keras.models.load_model('/Users/bunny/Documents/hyundairotem_aimodel/python_analysis/tensorflow/tf45_ex6_mac.keras')
target_height, target_width = model.input_shape[1:3]  # 모델이 기대하는 입력 이미지의 높이와 너비 추출

storage = FileSystemStorage()  # Django 기본 파일 시스템 스토리지 초기화


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

def main(request):
    context = {}  # 템플릿 렌더링에 사용할 데이터와 오류 메시지 저장

    if request.method == "POST":
        uploaded_file = request.FILES.get("image")  # 업로드된 이미지를 폼 데이터에서 추출

        if not uploaded_file:  # 업로드 파일이 없을 때 빈 전송 방지
            context["error"] = "이미지를 먼저 업로드 해주세요."
        else:
            try:
                extension = Path(uploaded_file.name).suffix.lower()  # 파일 확장자를 추출하고 소문자로 통일
                unique_name = f"{uuid.uuid4().hex}{extension}"  # 충돌을 피하기 위한 고유 파일명 생성
                saved_name = storage.save(f"uploads/{unique_name}", uploaded_file)  # 업로드 파일을 MEDIA_ROOT에 저장
                saved_url = storage.url(saved_name)  # 저장된 파일을 템플릿에서 접근할 수 있는 URL로 변환
                saved_path = storage.path(saved_name)  # 로컬 처리용 실제 파일 경로 확보

                pil_image = tf.keras.preprocessing.image.load_img(
                    saved_path,
                    target_size=(target_height, target_width),
                )  # 저장된 이미지를 PIL 객체로 로드하고 모델 입력 크기에 맞게 리사이즈
                image_array = tf.keras.preprocessing.image.img_to_array(pil_image)  # PIL 이미지를 NumPy 배열로 변환
                model_input = np.expand_dims(image_array / 255.0, axis=0)  # 픽셀을 정규화하고 배치 차원을 추가

                predictions = model.predict(model_input, verbose=0)[0]  # 추론을 수행하고 확률 벡터를 1차원으로 취득
                top_index = int(np.argmax(predictions))  # 가장 높은 확률을 가진 클래스 인덱스를 계산
                top_label = CLASSES[top_index]  # 최고 확률 인덱스를 실제 라벨 문자열로 변환
                top_confidence = float(predictions[top_index])  # 최고 확률 값을 float 형으로 변환

                top_indices = predictions.argsort()[-3:][::-1]  # 상위 세 개 확률에 해당하는 인덱스를 정렬해 추출
                top_classes = [
                    {
                        "label": CLASSES[index],
                        "confidence": float(predictions[index]),
                        "percentage": float(predictions[index] * 100),
                    }
                    for index in top_indices
                ]  # 상위 세 개 예측 클래스에 대해 라벨과 확률 정보를 구성

                image_for_plot = np.clip(image_array, 0, 255).astype(np.uint8)  # 시각화를 위해 0~255 범위의 8비트 이미지로 변환

                figure, axis = plt.subplots(figsize=(4, 4))  # 출력용 Matplotlib Figure와 Axis 생성
                axis.imshow(image_for_plot)  # 업로드된 이미지를 축에 출력
                axis.axis("off")  # 축 눈금을 숨겨 깔끔한 화면 구성
                figure.tight_layout(pad=0)  # 여백을 최소화하여 이미지에 딱 맞게 조정

                buffer = io.BytesIO()  # 렌더링된 이미지를 담을 메모리 버퍼 생성
                figure.savefig(buffer, format="png", bbox_inches="tight")  # 버퍼에 PNG 형식으로 저장
                plt.close(figure)  # Matplotlib 리소스를 즉시 해제
                buffer.seek(0)  # 버퍼 읽기를 위해 포인터를 처음으로 이동

                plot_name = storage.save(
                    f"plots/{uuid.uuid4().hex}.png", ContentFile(buffer.getvalue())
                )  # 렌더링된 플롯 이미지를 Django 스토리지에 저장
                plot_url = storage.url(plot_name)  # 템플릿에서 접근 가능한 플롯 이미지 URL 생성

                context.update(
                    {
                        "uploaded_image_url": saved_url,
                        "plot_url": plot_url,
                        "predicted_label": top_label,
                        "confidence": top_confidence,
                        "confidence_pct": top_confidence * 100,
                        "top_classes": top_classes,
                    }
                )  # 템플릿에 표시할 예측 결과와 가공 정보를 모두 context에 반영
            except Exception as exc:  # pragma: no cover - 예외 발생 시 UI에 메시지를 보여주기 위한 처리
                context["error"] = f"Unable to classify the image: {exc}"  # 예측 작업 중 문제를 사용자에게 전달

    return render(request, "main.html", context)
