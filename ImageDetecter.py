from flask import Flask, render_template, request, send_from_directory
import cv2
import os
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            detect_image(filepath)
            return send_from_directory(UPLOAD_FOLDER, file.filename)
    return render_template('index.html')

def detect_image(filepath):
    # YOLOv4 설정 파일 및 가중치 파일 경로
    model_cfg = 'yolov4/yolov4.cfg'
    model_weights = 'yolov4/yolov4.weights'
    class_names = 'yolov4/coco.names'
    
    # 클래스 이름 가져오기
    with open(class_names, 'r') as f:
        classes = f.read().strip().split('\n')

    # 각 클래스에 대한 랜덤 색상 생성
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

    # 네트워크 로딩
    net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # 이미지 로딩 및 전처리
    image = cv2.imread(filepath)
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # 객체 감지 실행
    #layer_names = net.getLayerNames()
    # output_layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    output_layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(output_layer_names)

    # ...

    # 감지 결과 처리 및 그리기
    h, w = image.shape[:2]
    boxes, confidences, class_ids = [], [], []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                box = obj[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # NMS 적용 후의 경계 상자 그리기
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 결과 이미지 저장
    cv2.imwrite(filepath, image)

if __name__ == '__main__':
    app.run(debug=True)
