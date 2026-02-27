from flask import Flask, request, send_file, render_template, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load YOLO model
model = YOLO("models/best_nano_111.pt")

# Create folder for detected images
os.makedirs("detected_fires", exist_ok=True)


# -------- Webpage --------
@app.route('/')
def index():
    return send_file('index.html')


# -------- Image Upload Inference --------
@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return {'error': 'No file uploaded'}, 400

    file = request.files['file']
    filepath = os.path.join("detected_fires", file.filename)
    file.save(filepath)

    # Run YOLO prediction
    results = model.predict(source=filepath, conf=0.35, iou=0.1)
    
    # Annotate image
    annotated_image = results[0].plot()
    output_path = os.path.join("detected_fires", f"output_{file.filename}")
    cv2.imwrite(output_path, annotated_image)

    return send_file(output_path, mimetype='image/png')


# -------- Webcam Real-Time Streaming --------
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference on the frame
        results = model.predict(source=frame, conf=0.35, iou=0.1)
        annotated_frame = results[0].plot()

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in multipart format for browser streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)