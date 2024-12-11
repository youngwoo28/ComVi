import os
import base64
from flask import Flask, request, jsonify, render_template, send_file
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
        # YOLO 모델 예측
        results = model(image_path)  # 모델 예측 수행
        print("모델 예측 완료")

        # YOLO 예측 결과를 처리
        if isinstance(results, list):
            results = results[0]  # 리스트의 첫 번째 결과 선택

        # 렌더링된 이미지를 가져오기
        rendered_image = results.plot()  # 최신 YOLO에서 plot() 메서드 사용
        print("렌더링 완료, 이미지 저장 중...")

        # numpy 배열을 Pillow 이미지로 변환
        img = Image.fromarray(rendered_image.astype(np.uint8))
        output_path = os.path.join("static", "result.jpg")
        img.save(output_path, format="JPEG")
        print("이미지 저장 완료:", output_path)

        return output_path
    
    except Exception as e:
        print("process_image 함수에서 예외 발생:", str(e))
        raise e

# API 엔드포인트: /predict
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            print("이미지 요청 수신됨")
            if 'image' not in request.files:
                print("이미지가 요청에 포함되지 않았습니다.")
                return jsonify({"error": "이미지를 업로드하세요"}), 400

            file = request.files['image']
            if file.filename == '':
                print("파일 이름이 비어 있습니다.")
                return jsonify({"error": "파일 이름이 비어 있습니다"}), 400

            # 이미지 저장 경로
            image_path = os.path.join("uploads", file.filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(image_path)

            print(f"이미지 저장 완료: {image_path}")

            # 이미지 판독
            result_image_path = process_image(image_path)
            print("이미지 판독 완료, 결과 반환 중...")

            return render_template('result.html', result_image=result_image_path)
        
        except Exception as e:
            print("API 핸들러에서 예외 발생:", str(e))
            import traceback
            traceback.print_exc()  # 상세한 예외 정보 출력
            return jsonify({"error": str(e)}), 500
        
        finally:
            # 업로드된 파일 삭제
            if 'image_path' in locals() and os.path.exists(image_path):
                os.remove(image_path)
                print(f"임시 파일 삭제 완료: {image_path}")

    return render_template('index.html')

# Flask 서버 실행
if __name__ == '__main__':
    app.run(debug=True)
