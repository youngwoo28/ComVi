import os
import base64
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
from flask_cors import CORS

# 환경 변수 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Flask 앱 생성
app = Flask(__name__)
CORS(app)  # CORS 활성화

# YOLO 모델 로드
model_path = "yolo11n.pt"  # 훈련된 YOLO 모델 경로
if not os.path.exists(model_path):
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
print(f"모델 로드 중... {model_path}")
model = YOLO(model_path)

# 이미지 처리 및 예측 함수
def process_image(image_path):
    try:
        print(f"이미지 처리 시작: {image_path}")
        results = model(image_path)  # 모델 예측 수행
        print(f"예측 결과: {results}")

        # YOLO 예측 결과 처리
        if isinstance(results, list) and len(results) > 0:
            results = results[0]  # 리스트의 첫 번째 결과 선택
        else:
            raise ValueError("모델이 결과를 반환하지 않았습니다.")

        # 렌더링된 이미지를 가져오기
        rendered_image = results.plot()
        img = Image.fromarray(rendered_image.astype(np.uint8))

        # 처리된 이미지를 저장
        output_path = os.path.join("uploads", "result.jpg")
        img.save(output_path, format="JPEG")
        print(f"이미지 저장 완료: {output_path}")

        return output_path
    except Exception as e:
        print(f"process_image 함수 오류: {str(e)}")
        raise e

# API 엔드포인트: /predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("POST 요청 수신")
        if 'image' not in request.files:
            raise ValueError("이미지 파일이 요청에 포함되지 않았습니다.")

        file = request.files['image']
        if file.filename == '':
            raise ValueError("업로드된 파일 이름이 없습니다.")

        # 이미지 저장 경로
        image_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(image_path)
        print(f"이미지 저장 성공: {image_path}")

        # 이미지 처리
        result_image_path = process_image(image_path)
        return jsonify({"image_url": f"uploads/{os.path.basename(result_image_path)}"})
    except Exception as e:
        print(f"predict 엔드포인트 오류: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # 업로드된 파일 삭제
        if 'image_path' in locals() and os.path.exists(image_path):
            os.remove(image_path)
            print(f"임시 파일 삭제 완료: {image_path}")

# 정적 파일 서빙
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# Flask 서버 실행
if __name__ == '__main__':
    app.run(debug=True)
