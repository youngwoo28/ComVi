import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import zipfile
import yaml
from ultralytics import YOLO

# 1. 데이터셋 다운로드 및 압축 해제
dataset_url = "https://universe.roboflow.com/ds/m9MDY9ooeS?key=c83nymrp0p"
dataset_zip = "GoalLine.zip"
dataset_dir = os.path.abspath("GoalLine_Data")  # 절대 경로로 설정

# 데이터셋 다운로드
if not os.path.exists(dataset_zip):
    print("데이터셋 다운로드 중...")
    os.system(f"wget -O {dataset_zip} {dataset_url}")

# 데이터셋 압축 해제
if not os.path.exists(dataset_dir):
    print("데이터셋 압축 해제 중...")
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

# 데이터셋 구조 디버깅
print("데이터셋 구조:")
for root, dirs, files in os.walk(dataset_dir):
    print(f"{root} -> 디렉토리: {dirs}, 파일: {files}")

# 2. data.yaml 파일 생성
data_yaml_path = os.path.join(dataset_dir, "data.yaml")
data = {
    'train': os.path.join(dataset_dir, 'train', 'images'),
    'val': os.path.join(dataset_dir, 'valid', 'images'),
    'test': os.path.join(dataset_dir, 'test', 'images'),
    'names': ['Ball', 'Football', 'Inside Goal Area', 'Line', 'Playing Area'],
    'nc': 5,
}

with open(data_yaml_path, 'w') as f:
    yaml.dump(data, f)

print("data.yaml 파일 생성 완료:")
with open(data_yaml_path, 'r') as f:
    print(yaml.safe_load(f))

# 3. YOLO 모델 불러오기
model_path = "yolo11n.pt"  # 사전 학습된 YOLO 모델 경로
print("YOLO 모델 로드 중...")
model = YOLO(model_path)

# 4. 모델 훈련
print("모델 훈련 시작...")
model.train(data=data_yaml_path, epochs=30, patience=5, imgsz=416)
print("모델 훈련 완료!")

# 5. 모델 저장
print("모델 TorchScript 형식으로 저장 중...")
model.export(format='torchscript')

# 6. 테스트 데이터에 대한 결과 확인
print("테스트 데이터 결과 확인 중...")
results = model(source=os.path.join(dataset_dir, 'test', 'images'), save=True)
print("테스트 완료, 결과 저장됨!")
