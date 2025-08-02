from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import base64
import time

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Eye landmark IDs
RIGHT_EYE = [33, 133, 159, 145]  # [outer, inner, top, bottom]
LEFT_EYE = [362, 263, 386, 374]

eye_data = []
step = 0
distances = [20, 40, 100]  # cm
start_time = time.time()
dur_per_step = 4

def get_eye_aspect_ratio(landmarks, w, h, eye_ids):
    p1 = np.array([landmarks[eye_ids[0]].x * w, landmarks[eye_ids[0]].y * h])
    p2 = np.array([landmarks[eye_ids[1]].x * w, landmarks[eye_ids[1]].y * h])
    p3 = np.array([landmarks[eye_ids[2]].x * w, landmarks[eye_ids[2]].y * h])
    p4 = np.array([landmarks[eye_ids[3]].x * w, landmarks[eye_ids[3]].y * h])
    hor = np.linalg.norm(p1 - p2)
    ver = np.linalg.norm(p3 - p4)
    return ver / hor if hor > 0 else 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global step, start_time, eye_data

    img_data = request.json['frame'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    message = f"Step {step+1} - Distance: {distances[step]}cm"

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0].landmark

        re_width = np.linalg.norm([
            face[RIGHT_EYE[0]].x - face[RIGHT_EYE[1]].x,
            face[RIGHT_EYE[0]].y - face[RIGHT_EYE[1]].y
        ])
        re_ar = get_eye_aspect_ratio(face, w, h, RIGHT_EYE)

        le_width = np.linalg.norm([
            face[LEFT_EYE[0]].x - face[LEFT_EYE[1]].x,
            face[LEFT_EYE[0]].y - face[LEFT_EYE[1]].y
        ])
        le_ar = get_eye_aspect_ratio(face, w, h, LEFT_EYE)

        avg_width = (re_width + le_width) / 2
        avg_ar = (re_ar + le_ar) / 2
        eye_data.append((step, avg_width, avg_ar, time.time()))

    if time.time() - start_time > dur_per_step:
        step += 1
        start_time = time.time()

    # When test ends
    if step >= len(distances):
        import pandas as pd
        df = pd.DataFrame(eye_data, columns=["step", "width", "aspect", "timestamp"])
        summary = df.groupby("step").agg({"width": "mean", "aspect": "mean"})

        if len(summary) == 3:
            close = summary.loc[0, "width"]
            far = summary.loc[2, "width"]

            if close > far:
                power = round(-1.8 * (close - far), 2)
                result = f"🔎 Nearsighted (Myopic) ~{power} D"
            elif far > close:
                power = round(1.5 * (far - close), 2)
                result = f"🔎 Farsighted (Hyperopic) ~+{power} D"
            else:
                result = f"✅ Focus consistent. ~0.00 D"
        else:
            result = "⚠️ Not enough data collected."

        # Reset for next run
        step = 0
        eye_data = []
        return jsonify({"done": True, "result": result})
    
    return jsonify({"done": False, "message": message})

if __name__ == '__main__':
    app.run(debug=True)
