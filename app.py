import streamlit as st
import cv2
import os
import glob
import csv
import time
from datetime import datetime
from PIL import Image

from pipeline.detector import VideoDetector
from pipeline.vlm_engine import EventExplainer
from pipeline.db_handler import DatabaseManager

st.set_page_config(page_title="Enterprise Surveillance Triage", layout="wide")

VIDEO_DIR = os.path.join("data", "sample_videos")
LABELS_CSV = os.path.join(VIDEO_DIR, "labels.csv")
EVENT_MAP = {"fall": "Person Down", "fight": "Altercation/Proximity"}


@st.cache_resource
def load_system():
    return VideoDetector(), EventExplainer(), DatabaseManager()


detector, explainer, db = load_system()


def list_videos():
    files = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        files += glob.glob(os.path.join(VIDEO_DIR, ext))
    return sorted(files)


def load_labels():
    labels = {}
    if not os.path.exists(LABELS_CSV):
        return labels
    with open(LABELS_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = (row.get("video") or "").strip()
            event = (row.get("event") or "").strip().lower()
            if not name:
                continue
            try:
                t = float(row.get("time_sec"))
            except (TypeError, ValueError):
                t = None
            labels.setdefault(name, [])
            if event in EVENT_MAP and t is not None:
                labels[name].append((event, t))
    return labels


def analyse_video(path, progress=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": "could not open the video"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    detector.reset()
    detector.fps = fps
    step = 1 if fps < 10 else 2
    fid, analysed, alerts = 0, 0, []
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fid += 1
        if fid % step:
            continue
        _, events = detector.detect_triggers(frame, fid, camera_id="EVAL")
        analysed += 1
        for e in events:
            alerts.append({"time": fid / fps, "type": e["type"],
                           "rule": e.get("reason", ""), "keyframe": e["keyframe"]})
        if progress and analysed % 5 == 0:
            progress(min(fid / total, 1.0))
    cap.release()
    return {"alerts": alerts, "fps": fps, "duration": total / fps,
            "analysed": analysed, "secs": max(time.time() - t0, 1e-3)}


def score_results(results, labels, tol):
    rows = []
    tot = {"truth": 0, "hit": 0, "false": 0, "clips": 0}
    for name, r in results.items():
        if "error" in r:
            rows.append({"Video": name, "Length": "-", "Detector speed": "-",
                         "Labelled events": "-", "Alerts": "-", "Verdict": r["error"]})
            continue
        alerts = r["alerts"]
        labelled = name in labels
        truths = labels.get(name, [])
        used, hits, missed = set(), 0, []
        for ev, t in truths:
            found = None
            for i, a in enumerate(alerts):
                if i not in used and a["type"] == EVENT_MAP[ev] and t - 1.0 <= a["time"] <= t + tol:
                    found = i
                    break
            if found is None:
                missed.append(f"{ev}@{t:.0f}s")
            else:
                used.add(found)
                hits += 1
        false_alarms = [a for i, a in enumerate(alerts) if i not in used] if labelled else []
        if labelled:
            tot["clips"] += 1
            tot["truth"] += len(truths)
            tot["hit"] += hits
            tot["false"] += len(false_alarms)
            verdict = f"{hits}/{len(truths)} caught, {len(false_alarms)} false alarm(s)"
            if missed:
                verdict += " - missed: " + ", ".join(missed)
        else:
            verdict = "not in labels.csv"
        rows.append({
            "Video": name,
            "Length": f"{r['duration']:.0f}s @ {r['fps']:.0f}fps",
            "Detector speed": f"{r['analysed'] / r['secs']:.1f} frames/s",
            "Labelled events": ", ".join(f"{e}@{t:.0f}s" for e, t in truths) or ("none" if labelled else "-"),
            "Alerts": "; ".join(f"{a['time']:.1f}s {a['type']} [{a['rule']}]" for a in alerts) or "none",
            "Verdict": verdict,
        })
    return rows, tot


mode = st.sidebar.radio("Mode", ["Live monitoring", "Evaluation (test clips)"])

st.title("🛡️ Enterprise AI Surveillance & Incident Triage")
st.caption("Two-tier cascade: YOLO pose triggers (fast) + Vision-Language description/filtering (smart) + dual-database search")

if mode == "Live monitoring":
    stride = st.sidebar.slider("Analyse every Nth frame", 1, 5, 2)

    video_files = list_videos()
    sources = {"Webcam (Device 0)": 0}
    for f in video_files:
        sources["🎞️ " + os.path.basename(f)] = f

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📹 Live Video Feed")
        choice = st.selectbox("Select Camera Feed", list(sources.keys()))
        run_stream = st.checkbox("Start Monitoring", value=False)
        feed_window = st.image([])
        status_alert = st.empty()

        if run_stream:
            src = sources[choice]
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                st.error(f"Could not open: {src}")
            detector.reset()
            detector.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_id = 0
            last_annotated = None

            try:
                while run_stream and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Video finished. Untick and tick 'Start Monitoring' to replay.")
                        break
                    frame_id += 1

                    if frame_id % stride == 0:
                        results, triggers = detector.detect_triggers(frame, frame_id, camera_id="CAM_01")
                        last_annotated = results.plot()

                        for trig in triggers:
                            status_alert.warning(f"Candidate: {trig['type']} - checking with VLM...")
                            explanation = explainer.analyze_event(trig["keyframe"], trig["type"], trig.get("box"))

                            if explanation["category"] != "Normal Activity":
                                status_alert.error(f"⚠️ {explanation['category']} (Confidence: {explanation['confidence']})")
                                db.insert_event({
                                    "event_id": f"EVT_{int(time.time() * 1000)}_{frame_id}",
                                    "camera_id": "CAM_01",
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "category": explanation["category"],
                                    "confidence": explanation["confidence"],
                                    "summary": explanation["summary"],
                                    "keyframe_path": trig["keyframe"],
                                })
                                st.toast(f"Incident Logged: {explanation['category']}")
                            else:
                                status_alert.info(f"{trig['type']} candidate rejected by VLM (false alarm filtered).")

                    shown = last_annotated if last_annotated is not None else frame
                    feed_window.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
            finally:
                cap.release()
                cv2.destroyAllWindows()
        else:
            status_alert.info("Camera inactive. Pick a feed and check 'Start Monitoring'.")

    with col2:
        st.subheader("🔍 Conversational Incident Query")
        query = st.text_input("Ask about logged incidents (e.g., 'someone collapsed on floor', 'physical fight'):")

        if st.button("Search Records") and query:
            with st.spinner("Searching semantic database..."):
                results = db.search_semantic(query, n_results=3)
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            if docs:
                for doc, meta in zip(docs, metas):
                    with st.expander(f"{meta.get('category')} - {meta.get('timestamp')}", expanded=True):
                        st.write(f"**AI Description:** {doc}")
                        st.write(f"**Camera:** {meta.get('camera_id')} | **Time:** {meta.get('timestamp')}")
                        kpath = meta.get("keyframe_path")
                        if kpath and os.path.exists(kpath):
                            st.image(Image.open(kpath), caption="Keyframe Evidence", width="stretch")
            else:
                st.info("No matching incidents found.")

        st.markdown("---")
        st.subheader("📊 Recent Incident Ledger (SQLite)")
        recent_rows = db.query_sql(
            "SELECT event_id, camera_id, timestamp, category, confidence FROM security_events "
            "WHERE category != 'Normal Activity' ORDER BY timestamp DESC LIMIT 5")
        if recent_rows:
            st.table([{"ID": r[0], "Camera": r[1], "Timestamp": r[2], "Category": r[3],
                       "Confidence": f"{r[4]:.2f}"} for r in recent_rows])
        else:
            st.write("No incidents currently logged.")

else:
    st.subheader("🧪 Evaluation on test clips")
    st.caption("Runs only the detector on each clip (no VLM, nothing is written to the ledger) "
               "and compares the alerts with the ground truth in data/sample_videos/labels.csv.")
    videos = list_videos()
    labels = load_labels()

    with st.expander("How to label clips (labels.csv)"):
        st.write("File location: `data/sample_videos/labels.csv`. `event` is fall, fight or none. "
                 "`time_sec` is the second the event happens (blank for none).")
        st.code("video,event,time_sec\ntest.mp4,fall,5\ntest2.mp4,fall,18\nwalking1.avi,none,", language="text")
        if not os.path.exists(LABELS_CSV):
            st.warning("labels.csv not found yet, so results can't be scored.")

    if not videos:
        st.warning("No videos found in data/sample_videos.")
    else:
        names = [os.path.basename(v) for v in videos]
        chosen = st.multiselect("Clips to test", names, default=names)
        tol = st.number_input("Count an alert as correct if it comes up to this many seconds after the labelled time",
                              min_value=1.0, max_value=15.0, value=3.0, step=0.5)

        if st.button("Run evaluation") and chosen:
            results_new = {}
            bar = st.progress(0.0, text="Starting...")
            for k, name in enumerate(chosen):
                def cb(p, k=k, name=name):
                    bar.progress(min((k + p) / len(chosen), 1.0),
                                 text=f"Analysing {name} ({k + 1}/{len(chosen)})")
                results_new[name] = analyse_video(os.path.join(VIDEO_DIR, name), cb)
            bar.progress(1.0, text="Done")
            st.session_state["eval_results"] = results_new

        results = st.session_state.get("eval_results")
        if results:
            rows, tot = score_results(results, labels, tol)
            c1, c2, c3 = st.columns(3)
            c1.metric("Events caught", f"{tot['hit']} / {tot['truth']}")
            c2.metric("False alarms", tot["false"])
            c3.metric("Labelled clips", tot["clips"])
            st.table(rows)

            st.markdown("#### Keyframes of each alert")
            for name, r in results.items():
                if "error" in r or not r["alerts"]:
                    continue
                with st.expander(f"{name} - {len(r['alerts'])} alert(s)"):
                    for a in r["alerts"]:
                        if os.path.exists(a["keyframe"]):
                            st.image(Image.open(a["keyframe"]),
                                     caption=f"{a['time']:.1f}s - {a['type']} [{a['rule']}]",
                                     width="stretch")