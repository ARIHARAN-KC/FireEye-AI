from flask import Flask, request, send_file, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
import time
from flask_mail import Mail, Message
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash

from src.db import users_collection

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# -------- MAIL CONFIGURATION --------
app.config['MAIL_SERVER'] = os.getenv("MAIL_SERVER")
app.config['MAIL_PORT'] = int(os.getenv("MAIL_PORT", 587))
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")

mail = Mail(app)

print("MAIL SERVER:", app.config['MAIL_SERVER'])
print("MAIL USER:", app.config['MAIL_USERNAME'])

# -------- Load YOLO Model --------
model = YOLO("models/best_nano_111.pt")

# Folder for detected images
os.makedirs("detected_fires", exist_ok=True)

# -------- USER SIGNUP --------
@app.route('/signup', methods=['POST'])
def signup():

    data = request.json

    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not name or not email or not password:
        return {"error": "Missing fields"}, 400

    if users_collection.find_one({"email": email}):
        return {"error": "User already exists"}, 400

    hashed_password = generate_password_hash(password)

    users_collection.insert_one({
        "name": name,
        "email": email,
        "password": hashed_password
    })

    return {"message": "User registered successfully"}

# -------- USER SIGNIN --------
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


# -------- FIRE ALERT EMAIL FUNCTION --------
def send_fire_alert(frame):

    try:
        users = users_collection.find({}, {"email": 1})

        filename = f"fire_{int(time.time())}.jpg"
        filepath = os.path.join("detected_fires", filename)

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
                msg.attach(filename, "image/jpeg", fp.read())

            mail.send(msg)

            print("🔥 Email sent to:", email)

    except Exception as e:
        print("Email sending failed:", e)


# -------- IMAGE FIRE DETECTION --------
@app.route('/detect_image', methods=['POST'])
def detect_image():

    if 'file' not in request.files:
        return {'error': 'No file uploaded'}, 400

    file = request.files['file']

    filepath = os.path.join("detected_fires", file.filename)
    file.save(filepath)

    results = model.predict(source=filepath, conf=0.35, iou=0.1)

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:

        frame = cv2.imread(filepath)
        send_fire_alert(frame)

    annotated_image = results[0].plot()

    output_path = os.path.join("detected_fires", f"output_{file.filename}")
    cv2.imwrite(output_path, annotated_image)

    return send_file(output_path, mimetype='image/png')


# -------- REALTIME WEBCAM FIRE DETECTION --------

last_alert_time = 0
ALERT_COOLDOWN = 60


def generate_frames():

    global last_alert_time

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model.predict(source=frame, conf=0.35, iou=0.1)

        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:

            current_time = time.time()

            if current_time - last_alert_time > ALERT_COOLDOWN:

                print("🔥 Fire detected! Sending alert...")

                send_fire_alert(frame)

                last_alert_time = current_time

        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# -------- TEST EMAIL ROUTE --------
@app.route('/test_email')
def test_email():

    try:

        msg = Message(
            subject="FireEye AI Test Email",
            sender=app.config["MAIL_USERNAME"],
            recipients=[app.config["MAIL_USERNAME"]]
        )

        msg.body = "Test email from FireEye AI system"

        mail.send(msg)

        return "Email sent successfully!"

    except Exception as e:
        return str(e)


# -------- WEB PAGES --------
@app.route('/')
def index():
    return send_file('index.html')


@app.route('/auth')
def auth():
    return send_file('auth.html')


@app.route('/how-it-works')
def how_it_works():
    return send_file('how_it_works.html')


# -------- RUN SERVER --------
if __name__ == '__main__':
    app.run(debug=True)