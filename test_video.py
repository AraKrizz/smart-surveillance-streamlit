import sys, os, cv2
from pipeline.detector import VideoDetector

path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_videos/test.mp4"
print("Video path:", os.path.abspath(path), "| exists:", os.path.exists(path))
cap = cv2.VideoCapture(path)
print("Opened:", cap.isOpened())
fps = cap.get(cv2.CAP_PROP_FPS) or 25
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"fps={fps:.1f} frames={total} length={total / fps:.1f}s")

det = VideoDetector()
det.fps = fps
step = 1 if fps < 10 else 2
per_sec = max(int(fps), 1)
fid = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    fid += 1
    if fid % step:
        continue
    res, events = det.detect_triggers(frame, fid)
    n = 0 if res.boxes is None else len(res.boxes)
    t = fid / fps
    if fid % per_sec < step or 16 <= t <= 24 or 49 <= t <= 57:
        info = []
        for pid, h in det.history.items():
            if h and h[-1]["frame"] == fid:
                s = h[-1]
                info.append(f"id{pid}: ratio={s['ratio_s']:.2f} angle={s['angle_s']}")
        print(f"t={t:5.1f}s people={n} {info}")
    for e in events:
        print(f"   >>> CANDIDATE at {t:.1f}s: {e['type']}  (rule: {e.get('reason')})")
print("done, frames read:", fid)