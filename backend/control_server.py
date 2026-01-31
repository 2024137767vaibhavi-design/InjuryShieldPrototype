# control_server.py
# Deploy-friendly FastAPI backend:
# - DOES NOT open webcam on server (Render has no webcam)
# - Frontend sends frames -> POST /process-frame
# - MediaPipe Pose -> classify exercise + check form
# - Writes live state to Firestore: postureLogs/latest
# - Logs wrong events to postureHistory

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import math
import json
import os

import firebase_admin
from firebase_admin import credentials, firestore

# -------------------- Firebase (SAFE for public repo) --------------------
# Put your Firebase service account JSON into Render Environment variable:
# FIREBASE_SERVICE_ACCOUNT_JSON = { ... full json ... }
#
# If you are testing locally and want to use serviceAccountKey.json,
# set USE_LOCAL_FIREBASE_KEY=1 in your local env and keep the file in backend folder.

db = None

def init_firebase():
    global db

    if firebase_admin._apps:
        db = firestore.client()
        return

    try:
        use_local = os.getenv("USE_LOCAL_FIREBASE_KEY", "0") == "1"

        if use_local and os.path.exists("serviceAccountKey.json"):
            #cred = credentials.Certificate("serviceAccountKey.json")
            #firebase_admin.initialize_app(cred)
            #db = firestore.client()
            print("✅ Firebase initialized using local serviceAccountKey.json")
            return

        firebase_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
        if not firebase_json:
            print("⚠️ Firebase not initialized: FIREBASE_SERVICE_ACCOUNT_JSON not set.")
            db = None
            return

        cred_dict = json.loads(firebase_json)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase initialized using FIREBASE_SERVICE_ACCOUNT_JSON")
    except Exception as e:
        print("⚠️ Firebase init failed:", e)
        db = None

init_firebase()

# -------------------- FastAPI --------------------
app = FastAPI()

# CORS: allow all for now (easy deployment). Lock later to your frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- MediaPipe --------------------
mp_pose = mp.solutions.pose

# -------------------- Geometry helpers --------------------
def angle3(a, b, c):
    """Angle ABC in degrees using numpy vectors for (x,y)."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def line_angle_deg(p1, p2):
    """
    Absolute angle of the line p1->p2 relative to horizontal.
    Bigger value => more vertical-ish/steeper.
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(math.degrees(math.atan2((y2 - y1), (x2 - x1))))

# -------------------- Exercise classification --------------------
def classify_exercise(lm):
    """
    Rule-based exercise guess (front camera works best).
    Returns one of: "Squat", "Deadlift", "Bicep Curl", "Shoulder Press"
    """
    l_sh = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
    r_sh = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
    l_el = (lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y)
    r_el = (lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y)
    l_wr = (lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y)
    r_wr = (lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y)

    l_hp = (lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y)
    r_hp = (lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y)
    l_kn = (lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y)
    r_kn = (lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y)
    l_an = (lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y)
    r_an = (lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y)

    l_elbow = angle3(l_sh, l_el, l_wr)
    r_elbow = angle3(r_sh, r_el, r_wr)
    l_knee = angle3(l_hp, l_kn, l_an)
    r_knee = angle3(r_hp, r_kn, r_an)

    # Shoulder press: wrists above shoulders (smaller y = higher)
    wrists_above_shoulders = (l_wr[1] < l_sh[1] - 0.03) or (r_wr[1] < r_sh[1] - 0.03)
    if wrists_above_shoulders:
        return "Shoulder Press"

    # Bicep curl: elbow flexed + wrists not overhead
    elbow_flexed = (l_elbow < 110) or (r_elbow < 110)
    wrists_not_overhead = (l_wr[1] > l_sh[1] - 0.01) and (r_wr[1] > r_sh[1] - 0.01)
    if elbow_flexed and wrists_not_overhead:
        return "Bicep Curl"

    # Deadlift vs squat:
    knees_straightish = (l_knee > 140) and (r_knee > 140)
    mid_sh = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    mid_hp = ((l_hp[0] + r_hp[0]) / 2, (l_hp[1] + r_hp[1]) / 2)
    torso_lean = line_angle_deg(mid_sh, mid_hp)

    if knees_straightish and torso_lean > 25:
        return "Deadlift"

    return "Squat"

# -------------------- Form checks --------------------
def check_form(exercise, lm):
    l_sh = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
    r_sh = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
    l_hp = (lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y)
    r_hp = (lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y)

    l_el = (lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y)
    r_el = (lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y)
    l_wr = (lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y)
    r_wr = (lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y)

    l_kn = (lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y)
    r_kn = (lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y)
    l_an = (lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y)
    r_an = (lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y)

    l_knee = angle3(l_hp, l_kn, l_an)
    r_knee = angle3(r_hp, r_kn, r_an)
    l_elbow = angle3(l_sh, l_el, l_wr)
    r_elbow = angle3(r_sh, r_el, r_wr)

    mid_sh = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    mid_hp = ((l_hp[0] + r_hp[0]) / 2, (l_hp[1] + r_hp[1]) / 2)
    torso_lean = line_angle_deg(mid_sh, mid_hp)

    status = "correct"
    issue = "—"

    if exercise == "Squat":
        if min(l_knee, r_knee) > 165:
            status, issue = "wrong", "Not squatting (legs too straight)"
        elif min(l_knee, r_knee) < 65:
            status, issue = "wrong", "Too deep / knee overbend"
        elif torso_lean > 55:
            status, issue = "wrong", "Back leaning too much"

    elif exercise == "Deadlift":
        if min(l_knee, r_knee) < 120:
            status, issue = "wrong", "Knees bending too much (looks like squat)"
        elif torso_lean < 20:
            status, issue = "wrong", "Not hinging (too upright)"
        elif torso_lean > 70:
            status, issue = "wrong", "Back angle too aggressive (risk)"

    elif exercise == "Bicep Curl":
        if (l_el[1] < l_sh[1] + 0.05) or (r_el[1] < r_sh[1] + 0.05):
            status, issue = "wrong", "Elbow lifted too high (cheating)"
        if (l_wr[1] < l_sh[1] - 0.03) or (r_wr[1] < r_sh[1] - 0.03):
            status, issue = "wrong", "Wrist too high (not curl form)"
        if max(l_elbow, r_elbow) > 175:
            status, issue = "wrong", "Arms too straight (no curl)"

    elif exercise == "Shoulder Press":
        wrists_above = (l_wr[1] < l_sh[1] - 0.03) or (r_wr[1] < r_sh[1] - 0.03)
        if not wrists_above:
            status, issue = "wrong", "Press not overhead enough"
        if torso_lean > 60:
            status, issue = "wrong", "Leaning too much while pressing"

    return status, issue

# -------------------- Firestore writer --------------------
def send_to_firebase(status, exercise, issue):
    if db is None:
        return  # Firebase not configured in environment

    try:
        db.collection("postureLogs").document("latest").set(
            {
                "status": status,
                "exercise": exercise,
                "issue": issue,
                "timestamp": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

        if status == "wrong":
            db.collection("postureHistory").add(
                {
                    "status": "wrong",
                    "exercise": exercise,
                    "issue": issue,
                    "timestamp": firestore.SERVER_TIMESTAMP,
                }
            )
    except Exception as e:
        print("⚠️ Firestore write failed (continuing):", e)

# -------------------- Simple health/status --------------------
@app.get("/")
def root():
    return {"ok": True, "msg": "Backend is running"}

@app.get("/status")
def api_status():
    # No server webcam mode in deployment
    return {"running": False, "mode": "frame-upload"}

# -------------------- Main endpoint: frontend sends frames --------------------
@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    data = await file.read()
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return {"ok": False, "msg": "Invalid image"}

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process pose
    with mp_pose.Pose(model_complexity=1) as pose:
        results = pose.process(frame_rgb)

    exercise = "Squat"
    status = "correct"
    issue = "—"

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        exercise = classify_exercise(lm)
        status, issue = check_form(exercise, lm)

    # Log to Firebase (if configured)
    send_to_firebase(status, exercise, issue)

    return {"ok": True, "exercise": exercise, "status": status, "issue": issue}
