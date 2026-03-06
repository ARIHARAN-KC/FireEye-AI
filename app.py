from flask import Flask, request, send_file, render_template, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
import io
import time
from flask_mail import Mail, Message
from src.config import Config
from PIL import Image
from src.db import users_collection
from werkzeug.security import generate_password_hash, check_password_hash
from src.notification_service import NotificationService

app = Flask(__name__)
CORS(app)

app.config.from_object(Config)
mail = Mail(app)

# Load YOLO model
model = YOLO("models/best_nano_111.pt")

# Create folder for detected images
os.makedirs("detected_fires", exist_ok=True)

# -------- User Signup --------
@app.route('/signup', methods=['POST'])
def signup():

    data = request.json

    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not name or not email or not password:
        return {"error": "Missing fields"}, 400

    # Check if user already exists
    if users_collection.find_one({"email": email}):
        return {"error": "User already exists"}, 400

    hashed_password = generate_password_hash(password)

    users_collection.insert_one({
        "name": name,
        "email": email,
        "password": hashed_password
    })

    return {"message": "User registered successfully"}

#---------send alert Notification is fire is detected---------
def send_fire_alert(frame):

    with app.app_context():

        users = users_collection.find({}, {"email": 1})

        filename = f"fire_{int(time.time())}.jpg"
        filepath = os.path.join("detected_fires", filename)

        # Save captured frame
        cv2.imwrite(filepath, frame)

        for user in users:

            email = user.get("email")

            msg = Message(
                subject="🔥 Fire Detected Alert",
                sender=app.config["MAIL_USERNAME"],
                recipients=[email]
            )

            msg.body = """
🔥 FireEye AI Alert

Fire detected in monitored area.

Please check immediately.

FireEye AI System
"""

            with open(filepath, "rb") as fp:
                msg.attach(
                    filename,
                    "image/jpeg",
                    fp.read()
                )

            mail.send(msg)

notification_service = NotificationService(app, mail, users_collection, Config)

# -------- User Signin --------
@app.route('/signin', methods=['POST'])
def signin():

    data = request.json

    email = data.get("email")
    password = data.get("password")

    user = users_collection.find_one({"email": email})

    if not user:
        return {"error": "User not found"}, 404

    if check_password_hash(user["password"], password):
        return {
            "message": "Login successful",
            "user": {
                "name": user["name"],
                "email": user["email"]
            }
        }

    return {"error": "Invalid password"}, 401

#---------Auth webpage---------
# Serve the authentication page
@app.route('/auth')
def auth():
    return send_file('auth.html')

#---------How It Works webpage---------
# Serve the How It Works page
@app.route('/how-it-works')
def how_it_works():
    return send_file('how_it_works.html')

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

    boxes = results[0].boxes

    # FIRE DETECTED → SEND EMAIL
    if boxes is not None and len(boxes) > 0:
        frame = cv2.imread(filepath)
        notification_service.send_fire_email(frame)

    annotated_image = results[0].plot()

    output_path = os.path.join("detected_fires", f"output_{file.filename}")
    cv2.imwrite(output_path, annotated_image)

    return send_file(output_path, mimetype='image/png')


# -------- Webcam Real-Time Streaming --------

last_alert_time = 0
ALERT_COOLDOWN = 60

def generate_frames():

    global last_alert_time

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Run YOLO detection
        results = model.predict(source=frame, conf=0.35, iou=0.1)

        boxes = results[0].boxes

        # Fire detected
        if boxes is not None and len(boxes) > 0:

            current_time = time.time()

            if current_time - last_alert_time > ALERT_COOLDOWN:

                send_fire_alert(frame)

                last_alert_time = current_time

        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)