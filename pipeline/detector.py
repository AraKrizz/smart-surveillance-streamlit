import cv2, os, time, math
from collections import deque, defaultdict
import numpy as np
from ultralytics import YOLO

DEBUG = False
IMGSZ = 1280
CONF = 0.25
MIN_TRACK_SAMPLES = 6

# ---- fall: lying pose ----
LYING_RATIO = 1.0
LYING_ANGLE = 55
LYING_ANGLE_RATIO = 0.7
JUMP_FACTOR = 2.5
JUMP_MIN_RATIO = 1.0
ONE_FRAME_RATIO = 1.2       # a single very wide box after being upright counts
UPRIGHT_ANGLE = 35
UPRIGHT_RATIO = 0.6
LIE_SECONDS = 0.4
TRANSITION_SEC = 4.0
FALL_COOLDOWN_SEC = 6.0
SMOOTH_N = 3

# ---- fall: fast drop / vanish (for occluded falls) ----
DROP_WINDOW_SEC = 1.2
DROP_MIN_FRAC = 0.35
DROP_SPEED = 1.0
DROP_SHRINK = 0.85
LOST_WINDOW_SEC = 1.5
EDGE_MARGIN = 0.04
CUTOFF_MARGIN = 0.02        # box touching the frame edge with no torso keypoints is ignored

# ---- fight ----
CLOSE_SEC = 1.0
ARM_SPEED = 2.0
BODY_SPEED = 1.2
SPEED_CAP = 6.0
ACTIVE_WINDOW_SEC = 2.0
ACTIVE_FRACTION = 0.5
MIN_KP_CONF = 0.5
FIGHT_COOLDOWN_SEC = 8.0


class VideoDetector:
    def __init__(self, model_name="yolov8s-pose.pt"):
        self.model = YOLO(model_name)
        self.keyframe_dir = os.path.join("data", "keyframes")
        os.makedirs(self.keyframe_dir, exist_ok=True)
        self.fps = 25.0
        self.reset()

    def reset(self):
        self.history = defaultdict(lambda: deque(maxlen=300))
        self.recent_upright = deque(maxlen=400)
        self.last_fall = -10**9
        self.last_fight = -10**9
        self.pair_close = {}
        self.lost_checked = set()
        self.tracks = {}
        self.next_id = 1
        self._last_frame = 0

    # ---------- helpers ----------
    def _assign_ids(self, xyxy, frame_id):
        """Simple tracker: match each box to the nearest previous box."""
        ids, used = [], set()
        for bb in xyxy:
            b = [float(v) for v in bb]
            best, best_score = None, 0.0
            for tid, (tb, tf) in self.tracks.items():
                if tid in used or frame_id - tf > 1.5 * self.fps:
                    continue
                iw = min(b[2], tb[2]) - max(b[0], tb[0])
                ih = min(b[3], tb[3]) - max(b[1], tb[1])
                inter = max(iw, 0) * max(ih, 0)
                union = ((b[2] - b[0]) * (b[3] - b[1])
                         + (tb[2] - tb[0]) * (tb[3] - tb[1]) - inter)
                iou = inter / union if union > 0 else 0.0
                d = math.hypot((b[0] + b[2]) / 2 - (tb[0] + tb[2]) / 2,
                               (b[1] + b[3]) / 2 - (tb[1] + tb[3]) / 2)
                close = d < 0.6 * max(b[3] - b[1], tb[3] - tb[1])
                score = iou if iou > 0.1 else (0.3 if close else 0.0)
                if score > best_score:
                    best, best_score = tid, score
            if best is None:
                best = self.next_id
                self.next_id += 1
            used.add(best)
            self.tracks[best] = (b, frame_id)
            ids.append(best)
        return ids

    def _torso_angle(self, kp):
        if min(kp[5][2], kp[6][2], kp[11][2], kp[12][2]) < 0.3:
            return None
        sh = ((kp[5][0] + kp[6][0]) / 2, (kp[5][1] + kp[6][1]) / 2)
        hp = ((kp[11][0] + kp[12][0]) / 2, (kp[11][1] + kp[12][1]) / 2)
        return math.degrees(math.atan2(abs(sh[0] - hp[0]), abs(sh[1] - hp[1]) + 1e-6))

    def _med(self, hist, key, n=SMOOTH_N):
        vals = [t[key] for t in list(hist)[-n:] if t[key] is not None]
        return float(np.median(vals)) if vals else None

    def _lying(self, s):
        r, a = s["ratio_s"], s["angle_s"]
        if r is None:
            return False
        if r > LYING_RATIO:
            return True
        if a is not None and a > LYING_ANGLE and r > LYING_ANGLE_RATIO:
            return True
        base = s.get("base")
        if base is not None and r > JUMP_MIN_RATIO and r > JUMP_FACTOR * base:
            return True
        return False

    def _upright(self, s):
        r, a = s["ratio_s"], s["angle_s"]
        if r is None:
            return False
        if a is not None:
            return a < UPRIGHT_ANGLE and r < UPRIGHT_RATIO
        return r < UPRIGHT_RATIO

    def _dropping(self, hist, s, frame_id):
        """Top edge of the box fell fast while the box shrank."""
        lo = frame_id - DROP_WINDOW_SEC * self.fps
        past = [t for t in hist if lo <= t["frame"] < frame_id]
        if len(past) < 2:
            return False
        first = past[0]
        ref_h = max(t["h"] for t in hist if t["frame"] > frame_id - 3 * self.fps)
        dt = max((frame_id - first["frame"]) / self.fps, 1e-3)
        drop = s["box"][1] - first["box"][1]
        return (drop > DROP_MIN_FRAC * ref_h
                and drop / dt / ref_h > DROP_SPEED
                and s["h"] < DROP_SHRINK * ref_h)

    def _close(self, sa, sb):
        ax1, ay1, ax2, ay2 = sa["box"]
        bx1, by1, bx2, by2 = sb["box"]
        gap_x = max(ax1, bx1) - min(ax2, bx2)
        gap_y = max(ay1, by1) - min(ay2, by2)
        avg_w = (sa["w"] + sb["w"]) / 2
        return gap_x < 0.25 * avg_w and gap_y < 0

    def _active(self, hist, frame_id):
        recent = [s for s in hist if s["frame"] > frame_id - ACTIVE_WINDOW_SEC * self.fps]
        if len(recent) < 6:
            return False
        fast = sum(1 for s in recent if s["sw"] > ARM_SPEED or s["sb"] > BODY_SPEED)
        return fast / len(recent) >= ACTIVE_FRACTION

    def _save(self, frame, tag, frame_id, camera_id):
        path = os.path.join(self.keyframe_dir, f"{camera_id}_{tag}_{int(time.time())}_{frame_id}.jpg")
        cv2.imwrite(path, frame)
        return path

    def _fire_fall(self, frame, frame_id, camera_id, box, events, reason):
        path = self._save(frame, "fall", frame_id, camera_id)
        events.append({"type": "Person Down", "keyframe": path,
                       "box": tuple(map(int, box)), "reason": reason})
        self.last_fall = frame_id

    # ---------- main ----------
    def detect_triggers(self, frame, frame_id, camera_id="CAM_01"):
        if frame_id < self._last_frame:
            self.reset()
        self._last_frame = frame_id
        H, W = frame.shape[:2]

        results = self.model.predict(frame, verbose=False,
                                     conf=CONF, imgsz=IMGSZ, classes=[0])[0]
        events, people = [], []
        boxes = results.boxes
        n = 0 if boxes is None else len(boxes)
        cur_ids = set()
        can_fall = frame_id - self.last_fall > FALL_COOLDOWN_SEC * self.fps

        if n > 0 and results.keypoints is not None:
            kpts = results.keypoints.data.cpu().numpy()
            ids = self._assign_ids(boxes.xyxy.cpu().numpy(), frame_id)

            for i, pid in enumerate(ids):
                cur_ids.add(pid)
                self.lost_checked.discard(pid)
                x1, y1, x2, y2 = map(float, boxes.xyxy[i])
                w, h = max(x2 - x1, 1), max(y2 - y1, 1)
                kp = kpts[i]
                s = {"frame": frame_id, "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                     "w": w, "h": h, "ratio": w / h, "angle": self._torso_angle(kp),
                     "sw": 0.0, "sb": 0.0, "box": (x1, y1, x2, y2),
                     "wrists": [kp[9][:2] if kp[9][2] > MIN_KP_CONF else None,
                                kp[10][:2] if kp[10][2] > MIN_KP_CONF else None]}
                hist = self.history[pid]
                if hist:
                    p = hist[-1]
                    dt = max((frame_id - p["frame"]) / self.fps, 1e-3)
                    s["sb"] = min(math.hypot(s["cx"] - p["cx"], s["cy"] - p["cy"]) / dt / h, SPEED_CAP)
                    sp = [math.hypot(*(s["wrists"][k] - p["wrists"][k])) / dt / h
                          for k in range(2)
                          if s["wrists"][k] is not None and p["wrists"][k] is not None]
                    s["sw"] = min(max(sp), SPEED_CAP) if sp else 0.0
                s["dropping"] = self._dropping(hist, s, frame_id) if hist else False
                hist.append(s)
                s["angle_s"] = self._med(hist, "angle")
                s["ratio_s"] = self._med(hist, "ratio")
                past = [t["ratio"] for t in hist if t["frame"] > frame_id - 4 * self.fps]
                s["base"] = min(past) if len(past) >= 4 else None
                people.append((pid, s))

                if self._upright(s):
                    self.recent_upright.append((frame_id, s["cx"], s["cy"], s["h"]))

                if DEBUG and frame_id % 15 == 0:
                    print(f"[dbg] f={frame_id} id={pid} ratio={s['ratio_s']:.2f} angle={s['angle_s']} "
                          f"lying={self._lying(s)} drop={s['dropping']}")

                if not can_fall or len(hist) < MIN_TRACK_SAMPLES:
                    continue

                # body cut off by the frame edge and no torso keypoints: can't judge a fall
                touches_edge = (x1 < CUTOFF_MARGIN * W or x2 > (1 - CUTOFF_MARGIN) * W or
                                y1 < CUTOFF_MARGIN * H or y2 > (1 - CUTOFF_MARGIN) * H)
                if touches_edge and s["angle"] is None:
                    continue

                # (a) fast drop while still visible
                if s["dropping"]:
                    self._fire_fall(frame, frame_id, camera_id, s["box"], events, "drop")
                    can_fall = False
                    continue

                # (b) lying pose
                if self._lying(s):
                    lie_start = frame_id - LIE_SECONDS * self.fps
                    recent = [t for t in hist if t["frame"] >= lie_start]
                    if any(self._lying(t) and t["ratio"] > ONE_FRAME_RATIO for t in recent) or \
                            (len(recent) >= 2 and all(self._lying(t) for t in recent)):
                        win_lo = frame_id - (LIE_SECONDS + TRANSITION_SEC) * self.fps
                        was_up = any(self._upright(t) for t in hist if win_lo <= t["frame"] <= lie_start)
                        if not was_up:
                            was_up = any(win_lo <= f <= lie_start and
                                         math.hypot(cx - s["cx"], cy - s["cy"]) < 2.0 * hh
                                         for (f, cx, cy, hh) in self.recent_upright)
                        if was_up:
                            self._fire_fall(frame, frame_id, camera_id, s["box"], events, "lying")
                            can_fall = False

        # (c) person vanished right after a fast drop (fell behind furniture)
        if can_fall:
            for pid, hist in list(self.history.items()):
                if pid in cur_ids or pid in self.lost_checked or len(hist) < MIN_TRACK_SAMPLES:
                    continue
                last = hist[-1]
                gap = frame_id - last["frame"]
                if gap <= 0 or gap > LOST_WINDOW_SEC * self.fps:
                    continue
                self.lost_checked.add(pid)
                x1, y1, x2, y2 = last["box"]
                near_edge = (x1 < EDGE_MARGIN * W or x2 > (1 - EDGE_MARGIN) * W or
                             y1 < EDGE_MARGIN * H or y2 > (1 - EDGE_MARGIN) * H)
                recently_dropping = any(t.get("dropping") for t in list(hist)[-3:])
                if recently_dropping and not near_edge:
                    self._fire_fall(frame, frame_id, camera_id, last["box"], events, "vanish")
                    break

        # ---------- FIGHT CANDIDATE ----------
        now_pairs, fired = set(), False
        for a in range(len(people)):
            for b in range(a + 1, len(people)):
                (ida, sa), (idb, sb_) = people[a], people[b]
                if len(self.history[ida]) < MIN_TRACK_SAMPLES or len(self.history[idb]) < MIN_TRACK_SAMPLES:
                    continue
                if not self._close(sa, sb_):
                    continue
                key = (min(ida, idb), max(ida, idb))
                now_pairs.add(key)
                start = self.pair_close.setdefault(key, frame_id)
                if (not fired and frame_id - start >= CLOSE_SEC * self.fps
                        and frame_id - self.last_fight > FIGHT_COOLDOWN_SEC * self.fps
                        and self._active(self.history[ida], frame_id)
                        and self._active(self.history[idb], frame_id)):
                    ax1, ay1, ax2, ay2 = sa["box"]
                    bx1, by1, bx2, by2 = sb_["box"]
                    path = self._save(frame, "fight", frame_id, camera_id)
                    events.append({"type": "Altercation/Proximity", "keyframe": path,
                                   "box": (int(min(ax1, bx1)), int(min(ay1, by1)),
                                           int(max(ax2, bx2)), int(max(ay2, by2))),
                                   "reason": "fight"})
                    self.last_fight = frame_id
                    fired = True
        self.pair_close = {k: v for k, v in self.pair_close.items() if k in now_pairs}

        return results, events