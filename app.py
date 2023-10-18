from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# 감정 분석을 위한 딥 러닝 모델 로드
emotion_model = load_model('emotion_model.h5')

# 웹페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html')

# 이미지 업로드 및 처리
@app.route('/upload', methods=['POST'])
def upload():
    # 업로드된 이미지를 받아옵니다.
    uploaded_image = request.files['image']

    # OpenCV를 사용하여 이미지를 읽고 얼굴을 인식합니다.
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # 얼굴 인식을 위한 OpenCV 코드 (예를 들어, Haar Cascade Classifier 사용)

    # 이미지를 모델에 맞게 전처리합니다.
    resized_image = cv2.resize(face_image, (48, 48))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    image_data = np.array(grayscale_image) / 255.0
    image_data = np.expand_dims(image_data, axis=0)
    image_data = image_data.reshape((1, 48, 48, 1))

    # 감정 분석 모델로 감정을 예측합니다.
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion_prediction = emotion_model.predict(image_data)
    emotion_index = np.argmax(emotion_prediction)
    predicted_emotion = emotion_labels[emotion_index]

    # 결과를 웹페이지로 반환합니다.
    return f"감정: {predicted_emotion}"

if __name__ == '__main__':
    app.run(debug=True)
