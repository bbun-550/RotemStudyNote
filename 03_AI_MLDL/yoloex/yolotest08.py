# 이미지 세그멘테이션
import os, cv2, numpy as np
from ultralytics import YOLO

IMG_PATH = 'yoloex/image_storage/stray_pet_detection3.jpg' # stray_pet_detection3.jpg / image1.jpg
OUT_DIR = 'yoloex/seg_out'
os.makedirs(OUT_DIR, exist_ok=True)

im = cv2.imread(IMG_PATH)
# 이미지가 제대로 로드되지 않았다면 즉시 중단하고 어떤 경로에서 실패했는지 알려줌
assert im is not None, f"이미지 읽기 실패 : {IMG_PATH}"

H, W = im.shape[:2]    # (127, 287, 3)
print(H, W)

# model
model = YOLO('yolov8n-seg.pt')
res = model(im)[0]
print(res)    # boxes, masks, names, array ...

cv2.imwrite(os.path.join(OUT_DIR, 'anno2.jpg'), res.plot())
# res.plot() : 원본 이미지 위에 바운딩 박스, 레이블, 신뢰도, 세그멘테이션마스크를 한번에 그려서 BGR 이미지로 반환(눈으로 확인하는 용도)

# 마스크가 없으면 종료
if res.masks is None or len(res.masks.data) == 0:
    print('마스크가 없어요')
    raise SystemExit

m_small = res.masks.data
# PyTorch 텐서 형태인 마스크를 (N,H,W) numpy 배열로 변환해 사용할 준비를 한다

# 각 객체 마스크를 원본 이미지 크기로 리사이즈하고 0/1로 이진화한 뒤 묶어준다
masks = np.stack(
    [
        cv2.resize(
            m.cpu().numpy().astype(np.float32),
            (W, H),
            interpolation=cv2.INTER_NEAREST
        ) > 0.5
        for m in m_small
    ],
    axis=0
)
print(masks)

# 세그 전 단계 : 마스크 프리뷰
# 마스크가 같은 위치 픽셀에 대해 객체 중 하나라도 1(True)이면 N개 마스크를 OR 연산으로 합친다
mask_union = (masks.any(axis=0).astype(np.uint) * 255)    # 모든 객체 마스크를 합쳐 단일 바이너리 마스크 생성(Bool→0/255 이미지). 흑백
cv2.imwrite(os.path.join(OUT_DIR, 'mask_preview.jpg'), mask_union)

# 최종 세그멘테이션 : 컬러 오버레이 + 외곽선
def color(i):
    return ((37 * i) % 256, (17 * i) % 256, (9 * i) % 256)    # BGR 랜덤하게 다른 색이 나올 수 있게. 
# 256 곱하는 이유는 8비트 이미지 채널이 0~255 값을 쓰기 때문에, 범위를 맞춰 0~255 내에서 반복되도록 한다


final = im.copy()    # 직접 원본에 덧칠하지 않고 안전하게 복사본에서 작업
blend = np.zeros_like(im)    # 오버레이 색 채우기 캔버스

# 컬러 오버레이(blend)는 객체 내부를 색칠하고 경계선을 그리는 두가지 작업을 for문에서 한다
for i, m in enumerate(masks):
    blend[m] = color(i)    # 마스크 영역에 칠해질 고유 색 채우기
    
    # RETR_EXTERNAL : 바깥 쪽 외곽선 만
    # CHAIN_APPROX_SIMPLE : 꼭짓점 단순화
    cnts, _ = cv2.findContours(    # 마스크 외곽선 자료 추출
        (m.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )    # 0/1 -> 0 ~ 255 이진화    / 가장 바깥 쪽 외곽선만 / 꼭짓점 단순화
    
    cv2.drawContours(final, cnts, -1, (255,255,255), 2, cv2.LINE_AA)    # cnts : 외곽선 리스트
    
# 반투명 합성
final = cv2.addWeighted(final, 1.0, blend, 0.45, 0.0)  # source1, alpha(1이미지 가중치), source2, beta(2이미지 가중치), gama(보정값)

cv2.imwrite(os.path.join(OUT_DIR, 'final_preview2.jpg'), final)

cv2.imshow('final segmentation', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
