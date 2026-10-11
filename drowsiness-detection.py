import time
import queue
import subprocess
import threading
import winsound

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# ---------- settings (one nam wenas karanna) ----------
CALIBRATION_SECONDS = 5   # palaweni seconds 5: eyes open, screen eka balan inna
EAR_RATIO = 0.75          # eyes close kiyala hithanne baseline eke 75% ta wada adu nam
CLOSED_SECONDS = 1.5      # eyes meka welawak close nam alert

PITCH_LIMIT = 20          # yatata / udata (nod)
ROLL_LIMIT = 20           # padawata (tilt)
HEAD_SECONDS = 1.0        # nod / tilt meka welawak thibboth alert

YAW_LIMIT = 25            # wamata / dakunata turn wela inna degrees
YAW_SECONDS = 30          # LEFT/RIGHT turn eka meka welawak thibboth alert

NO_FACE_SECONDS = 2.0     # face eka meka welawak penne nathnam alert

PHONE_SECONDS = 180       # phone eka 3 minutes penawa nam alert
PHONE_GRACE = 3.0         # phone eka tikak welawak nopenunath count eka reset wenne na
PHONE_CONF = 0.35         # phone detect confidence
DETECT_EVERY = 4          # frames 4k walata ekak phone check (PC eka slow nam 6-8)

MAR_LIMIT = 0.6           # mouth eka meka wadi open nam yawn ekak kiyala hithanawa
YAWN_SECONDS = 1.5        # mouth open eka meka welawak thibboth yawn ekak
YAWN_LIMIT = 3            # yawns meka gaana unama break warning
YAWN_WINDOW = 300         # ...meka seconds athule (300s = minutes 5)

ALERT_COOLDOWN = 4.0      # voice alert deka athare seconds
# ------------------------------------------------------

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
POSE_IDS = [1, 152, 33, 263, 61, 291]
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

# ---------- voice + beep (Windows voice, hama paaratama wada karanawa) ----------
speech_q = queue.Queue(maxsize=1)


def speak_windows(text):
    ps = ("Add-Type -AssemblyName System.Speech; "
          "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          "$s.Rate = 1; "
          f"$s.Speak('{text}')")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   creationflags=subprocess.CREATE_NO_WINDOW,
                   timeout=20)


def speaker_worker():
    while True:
        freq, text = speech_q.get()
        try:
            winsound.Beep(freq, 400)
            speak_windows(text)
        except Exception as e:
            print("Voice error:", e)


threading.Thread(target=speaker_worker, daemon=True).start()


def alert(freq, text):
    print("ALERT:", text)
    try:
        speech_q.put_nowait((freq, text))
    except queue.Full:
        pass   # kalin alert ekak kiyawenawa, eka nawaththanne na


VOICE = {
    "EYES CLOSED": "Wake up! Your eyes are closed.",
    "HEAD NOD/TILT": "Wake up! Keep your head up.",
    "HEAD TURNED": "Please face the screen.",
    "FACE NOT VISIBLE": "I can not see your face.",
    "PHONE USE": "Put the phone down and focus.",
    "TIRED": "You look tired. Please take a short break.",
}


def get_ear(pts, idx):
    p = [pts[i] for i in idx]
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return v / (2.0 * h + 1e-6)


def get_mar(pts):
    """Mouth aspect ratio: mouth height / mouth width."""
    v = np.linalg.norm(pts[13] - pts[14])
    h = np.linalg.norm(pts[78] - pts[308])
    return v / (h + 1e-6)


def head_pose(pts, w, h):
    """Returns (pitch, yaw, roll) in degrees, or None."""
    img_pts = np.array([pts[i] for i in POSE_IDS], dtype=np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, img_pts, cam, np.zeros((4, 1)))
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    a = cv2.RQDecomp3x3(rmat)[0]
    return np.array([a[0], a[1], a[2]], dtype=np.float64)


def wrap(a):
    return (a + 180) % 360 - 180


cap = cv2.VideoCapture(0)
mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
yolo = YOLO("yolov8n.pt")

start = time.time()
ear_samples, pose_samples = [], []
ear_thresh, base_pose = None, None
closed_since, head_since, no_face_since, last_alert = None, None, None, 0.0
yaw_since, last_yaw_bad = None, False
phone_since, last_phone_seen, phone_boxes = None, 0.0, []
mouth_since, yawn_counted, yawn_times = None, False, []
yawn_msg, yawn_msg_until = "", 0.0
frame_count = 0

print("Keys:  q = quit   |   t = test voice")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    now = time.time()
    frame_count += 1
    status, color = "No face", (0, 165, 255)
    reason, alert_freq = None, 1000

    # ---------- phone detection ----------
    if frame_count % DETECT_EVERY == 0:
        r = yolo(frame, verbose=False, classes=[67], conf=PHONE_CONF)[0]
        phone_boxes = [tuple(map(int, b.xyxy[0].tolist())) for b in r.boxes]
        if phone_boxes:
            last_phone_seen = now
            if phone_since is None:
                phone_since = now

    if phone_since is not None and now - last_phone_seen > PHONE_GRACE:
        phone_since = None

    phone_active = phone_since is not None
    phone_secs = (now - phone_since) if phone_active else 0

    for (x1, y1, x2, y2) in phone_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, "PHONE", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    if phone_active and phone_secs >= PHONE_SECONDS:
        reason, alert_freq = "PHONE USE", 1500

    # ---------- face / eyes / head / mouth ----------
    res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.multi_face_landmarks:
        no_face_since = None
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm])
        ear = (get_ear(pts, RIGHT_EYE) + get_ear(pts, LEFT_EYE)) / 2
        mar = get_mar(pts)
        pose = head_pose(pts, w, h)

        for i in RIGHT_EYE + LEFT_EYE:
            cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 2, (0, 255, 0), -1)
        for i in (13, 14, 78, 308):
            cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 2, (0, 255, 255), -1)

        if ear_thresh is None:
            # ---- calibration ----
            ear_samples.append(ear)
            if pose is not None:
                pose_samples.append(pose)
            left = CALIBRATION_SECONDS - (now - start)
            status, color = f"Calibrating {max(left, 0):.0f}s - eyes open, mouth closed", (255, 255, 0)
            if left <= 0 and len(ear_samples) > 10 and len(pose_samples) > 10:
                ear_thresh = float(np.median(ear_samples)) * EAR_RATIO
                base_pose = np.median(np.array(pose_samples), axis=0)
        else:
            # ---- eyes closed ----
            if ear < ear_thresh:
                closed_since = closed_since or now
                if now - closed_since >= CLOSED_SECONDS:
                    reason = reason or "EYES CLOSED"
            else:
                closed_since = None

            # ---- yawn detection ----
            if mar > MAR_LIMIT:
                mouth_since = mouth_since or now
                if now - mouth_since >= YAWN_SECONDS and not yawn_counted:
                    yawn_counted = True
                    yawn_times = [t for t in yawn_times if now - t <= YAWN_WINDOW]
                    yawn_times.append(now)
                    yawn_msg = f"Yawn detected ({len(yawn_times)}/{YAWN_LIMIT})"
                    yawn_msg_until = now + 4
                    if len(yawn_times) >= YAWN_LIMIT:
                        yawn_msg = "You look tired - take a break!"
                        yawn_msg_until = now + 8
                        alert(700, VOICE["TIRED"])
                        yawn_times = []
            else:
                mouth_since = None
                yawn_counted = False

            if pose is not None:
                d = np.array([wrap(pose[i] - base_pose[i]) for i in range(3)])

                # ---- nod / tilt (phone baladdi nawatthanawa) ----
                nod_tilt_bad = abs(d[0]) > PITCH_LIMIT or abs(d[2]) > ROLL_LIMIT
                if nod_tilt_bad and not phone_active:
                    head_since = head_since or now
                    if now - head_since >= HEAD_SECONDS:
                        reason = reason or "HEAD NOD/TILT"
                else:
                    head_since = None

                # ---- left / right turn ----
                yaw_bad = abs(d[1]) > YAW_LIMIT
                last_yaw_bad = yaw_bad
                if yaw_bad:
                    yaw_since = yaw_since or now
                    if now - yaw_since >= YAW_SECONDS:
                        reason = reason or "HEAD TURNED"
                else:
                    yaw_since = None

                cv2.putText(frame,
                            f"Up/Down {d[0]:+.0f}  Left/Right {d[1]:+.0f}  Tilt {d[2]:+.0f}",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.putText(frame, f"EAR {ear:.2f} (limit {ear_thresh:.2f})   MAR {mar:.2f} (limit {MAR_LIMIT})",
                        (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if not reason:
                status, color = "Alert", (0, 200, 0)
    else:
        closed_since = head_since = mouth_since = None
        if ear_thresh is not None:
            if last_yaw_bad and yaw_since is not None:
                if now - yaw_since >= YAW_SECONDS:
                    reason = reason or "HEAD TURNED"
                status, color = "Head turned away", (0, 165, 255)
            else:
                yaw_since = None
                if not phone_active:
                    no_face_since = no_face_since or now
                    if now - no_face_since >= NO_FACE_SECONDS:
                        reason = reason or "FACE NOT VISIBLE"

    # ---------- timers display ----------
    if phone_active:
        cv2.putText(frame, f"Phone: {phone_secs:.0f}s / {PHONE_SECONDS}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    if yaw_since is not None:
        cv2.putText(frame, f"Turned: {now - yaw_since:.0f}s / {YAW_SECONDS}s", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    if now < yawn_msg_until:
        cv2.putText(frame, yawn_msg, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # ---------- alert (voice + beep) ----------
    if reason:
        status, color = f"WAKE UP! ({reason})", (0, 0, 255)
        if now - last_alert > ALERT_COOLDOWN:
            alert(alert_freq, VOICE.get(reason, "Wake up!"))
            last_alert = now

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Drowsiness Detector (q = quit, t = test voice)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("t"):
        alert(1000, "Voice test. Wake up!")

cap.release()
cv2.destroyAllWindows()
