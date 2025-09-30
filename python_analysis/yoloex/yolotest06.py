# YOLO 탐지 결과를 이미지별로 정리해서 CSV로 저장한다
# 이어서 CSV를 읽어 DataFrame에 담아서 실습 진행한다

import os
import pandas as pd
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
img_dir = 'yoloex/image_storage'
img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.webp','.png','.jpeg'))]
# print(img_paths)

records = []

for path in img_paths:
    results = model(path, conf=0.25, verbose=False)[0]
    boxes = results.boxes
    names = results.names
    # print(boxes, names)
    
    if len(boxes) == 0:
        records.append({
            'image':os.path.basename(path),
            'object_count':0,
            'classes':'',
            'avg_confidence':0.0
        })
        continue
    
    cls_ids = boxes.cls.cpu().numpy().astype(int)    # numpy는 CPU 메모리에서만 동작한다
    # print(cls_ids)
    confs = boxes.conf.cpu().numpy()
    # print(confs)
    classes = [names[i] for i in cls_ids]
    # print(classes)
    avg_conf = float(confs.mean())
    
    records.append({
        'image':os.path.basename(path),
        'object_count':len(cls_ids),
        'classes':','.join(sorted(set(classes))),
        'avg_confidence':round(avg_conf, 3)
    })
    
# records -> DataFrame -> CSV
df = pd.DataFrame(records)
# print(df)
df.to_csv('yoloex/yolotest6.csv', index=False, encoding='utf-8-sig')
    
    
mydf = pd.read_csv('yoloex/yolotest6.csv')
num_images = len(mydf)
total_objects = mydf['object_count'].sum()
print(f'total_objects :{total_objects}')

# 전체 신뢰도 평균
overall_avg_conf = df.loc[df['avg_confidence'] > 0, 'avg_confidence'].mean() if total_objects > 0 else 0.0

# 클래스별 등장 빈도
class_counts = {}
for cls_str in df['classes']:
    if cls_str:
        for c in cls_str.split(','):    # 여러 클래스를 콤마를 구분되어 있다
            class_counts[c] = class_counts.get(c, 0) + 1
            

print("YOLO Detection Summary")
print(f"총 이미지 수     : {num_images}")
print(f"총 탐지 객체 수  : {total_objects}")
print(f"전체 신뢰도 평균 : {overall_avg_conf}")
print("\n클래스별 등장 이미지 횟수:")
for k, v in class_counts.items():
    print(f"{k} : {v}")



