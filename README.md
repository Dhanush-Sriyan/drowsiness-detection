# Drowsiness Detection System

A real-time drowsiness and distraction monitor for drivers and students during long study sessions. It uses a normal webcam and warns the user with a beep and a spoken alert.

Built with **Python, OpenCV, MediaPipe and YOLOv8**. Everything runs locally; no video is saved or uploaded.

> Screenshot / demo GIF: add `demo.gif` to the repo and uncomment the line below.
> <!-- ![demo](demo.gif) -->

## Features

| Feature | How it works | Alert |
|---|---|---|
| Eye closure | Eye Aspect Ratio (EAR) from 468 MediaPipe face landmarks | Eyes closed for 1.5 s |
| Head position | Head pose (pitch / yaw / roll) estimated with `solvePnP` | Nod or tilt for 1 s, turned left/right for 30 s |
| Yawn detection | Mouth Aspect Ratio (MAR) | 3 yawns in 5 minutes triggers a break reminder |
| Phone detection | YOLOv8 "cell phone" class | Phone visible for 3 minutes |
| Face missing | No face detected | Face not visible for 2 s |
| Voice + beep alerts | Windows speech synthesis, non-blocking | Different message per alert type |
| Auto-calibration | First 5 s measures the user's own open-eye EAR and neutral head pose | Works for different faces and camera angles |

## How it works

1. **Calibration (5 s):** the user looks at the screen with eyes open. The median EAR and head pose become the personal baseline.
2. **Monitoring:** every frame, EAR, MAR and head angles are compared with the baseline. Each condition must last longer than its time limit, so normal blinks and quick glances do not trigger alerts.
3. **Alerts:** a beep plus a voice message run in a background thread so the video never freezes.

## Installation

Requires **Windows** and **Python 3.11** (the voice alert uses `winsound` and Windows speech).

```bash
git clone https://github.com/YOUR-USERNAME/drowsiness-detection.git
cd drowsiness-detection
py -3.11 -m pip install -r requirements.txt
```

## Run

```bash
py -3.11 Drowsiness & Distraction Detection System.py
```

- Keep your eyes open and look at the screen for the first 5 seconds.
- `t` = test the voice, `q` = quit.
- The YOLO model (`yolov8n.pt`) downloads automatically on the first run.

## Settings

All thresholds are at the top of `Webcam.py`:

```python
EAR_RATIO = 0.75       # eye closure sensitivity
CLOSED_SECONDS = 1.5   # eyes closed time before alert
YAW_SECONDS = 30       # head turned left/right time before alert
PHONE_SECONDS = 180    # phone visible time before alert
MAR_LIMIT = 0.6        # yawn sensitivity
```

## Limitations

- Good, even lighting and a camera at eye level give the best results.
- Glasses with strong reflections can reduce eye-closure accuracy.
- This is a learning project, **not a certified safety device**. Do not rely on it for real driving.

## Future work

- Session log (CSV) and a focus-score dashboard
- PERCLOS metric
- Telegram notification to a phone
- Cross-platform voice alerts (macOS / Linux)

## Tech stack

Python · OpenCV · MediaPipe Face Mesh · NumPy · Ultralytics YOLOv8

## License

MIT
