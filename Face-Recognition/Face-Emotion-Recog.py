
import time
import cv2
import numpy as np
import psutil
import traceback

from threading import Thread, Event
from queue import Queue, Empty
from datetime import datetime

from insightface.app import FaceAnalysis

from model import db, FaceEmbedding512, EntryLog
from Emotion_Detection import (
    detect_emotion,
    close_emotion_detector,
)

import os
import sys
from pathlib import Path

# Load NVIDIA CUDA and cuDNN DLL directories
_dll_handles = []

_nvidia_dir = (
    Path(sys.prefix)
    / "Lib"
    / "site-packages"
    / "nvidia"
)

for _sub in (
    "cudnn/bin",
    "cublas/bin",
    "cuda_runtime/bin",
    "cuda_nvrtc/bin",
    "cuda_nvjitlink/bin",
):
    _dll_path = _nvidia_dir / _sub

    if _dll_path.exists():
        os.environ["PATH"] = (
            str(_dll_path)
            + os.pathsep
            + os.environ.get("PATH", "")
        )
        _dll_handles.append(
            os.add_dll_directory(str(_dll_path))
        )

import onnxruntime as ort

ort.preload_dlls(directory="")




# ==================================================
# Settings
# ==================================================

WINDOW_NAME = "Smart Entry System"

# Initial cosine similarity threshold.
# Must be calibrated using real camera footage.
SIMILARITY_THRESHOLD = 0.45

DETECTION_INTERVAL = 5
RECOGNITION_INTERVAL = 5
EMOTION_INTERVAL = 5

SCALE = 0.5
ENTRY_COOLDOWN = 30
'''
RTSP_URL = os.getenv(
    "SMART_ENTRY_RTSP_URL",
    "rtsp://root:root@192.168.10.176:554/axis-media/media.amp",
).strip()
'''

RTSP_URL = os.getenv("SMART_ENTRY_RTSP_URL", "").strip()

if not RTSP_URL:
    raise RuntimeError(
        "SMART_ENTRY_RTSP_URL is missing. "
        "Configure the camera URL in smart-entry.env."
    )

# CPU only until the cuDNN issue is resolved.
FACE_APP = FaceAnalysis(
    name="buffalo_l",
    #providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    providers=["CPUExecutionProvider"]
)

FACE_APP.prepare(
    ctx_id=0,
    det_size=(640, 640),
)


# ==================================================
# Queues and Stop Event
# ==================================================

frame_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=1)
stop_event = Event()


# ==================================================
# Load 512D Embeddings from Database
# ==================================================

known_embeddings = []
known_names = []
known_person_ids = []

for item in FaceEmbedding512.select():

    embedding = np.asarray(
        item.embedding,
        dtype=np.float32,
    )

    if embedding.shape != (512,):
        print(
            f"Skipping invalid embedding: "
            f"{item.image_name}, shape={embedding.shape}"
        )
        continue

    known_embeddings.append(embedding)
    known_names.append(item.person.name)
    known_person_ids.append(item.person.id)


if known_embeddings:
    known_embeddings = np.stack(known_embeddings)
else:
    known_embeddings = np.empty(
        (0, 512),
        dtype=np.float32,
    )

print(
    f"Loaded {len(known_embeddings)} "
    f"InsightFace embeddings from database"
)


# Reload embeddings periodically so people enrolled from the web panel become
# recognizable without restarting this process.
from threading import Lock
gallery_lock = Lock()
last_gallery_refresh = 0.0
GALLERY_REFRESH_SECONDS = 5.0


def refresh_known_gallery(force=False):
    global known_embeddings, known_names, known_person_ids, last_gallery_refresh
    now = time.monotonic()
    if not force and now - last_gallery_refresh < GALLERY_REFRESH_SECONDS:
        return
    with gallery_lock:
        now = time.monotonic()
        if not force and now - last_gallery_refresh < GALLERY_REFRESH_SECONDS:
            return
        embeddings = []
        names = []
        person_ids = []
        try:
            with db.connection_context():
                for item in FaceEmbedding512.select():
                    vector = np.asarray(item.embedding, dtype=np.float32)
                    if vector.shape != (512,):
                        continue
                    embeddings.append(vector)
                    names.append(item.person.name)
                    person_ids.append(item.person.id)
        except Exception as error:
            print(f"Gallery refresh failed; keeping previous gallery: {error}", flush=True)
            return

        new_embeddings = (
            np.stack(embeddings).astype(np.float32)
            if embeddings else np.empty((0, 512), dtype=np.float32)
        )
        known_embeddings = new_embeddings
        known_names = names
        known_person_ids = person_ids
        last_gallery_refresh = now
        print(f"Refreshed recognition gallery: {len(known_embeddings)} embeddings", flush=True)


# ==================================================
# Face Recognition using Cosine Similarity
# ==================================================

def recognize_face(embedding):

    refresh_known_gallery()

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    gallery_embeddings = known_embeddings
    gallery_names = known_names
    gallery_person_ids = known_person_ids

    if (
        embedding.shape != (512,)
        or len(gallery_embeddings) == 0
    ):
        return {
            "name": "Unknown",
            "distance": None,
            "person_id": None,
        }

    # Normalize vectors before cosine comparison.
    query_norm = np.linalg.norm(embedding)

    if query_norm == 0:
        return {
            "name": "Unknown",
            "distance": None,
            "person_id": None,
        }

    query = embedding / query_norm

    known_norms = np.linalg.norm(
        gallery_embeddings,
        axis=1,
        keepdims=True,
    )

    normalized_known = gallery_embeddings / np.maximum(
        known_norms,
        1e-12,
    )

    similarities = normalized_known @ query

    best_index = int(np.argmax(similarities))
    best_similarity = float(similarities[best_index])

    if best_similarity < SIMILARITY_THRESHOLD:
        return {
            "name": "Unknown",
            "distance": 1.0 - best_similarity,
            "person_id": None,
        }

    return {
        "name": gallery_names[best_index],
        "distance": 1.0 - best_similarity,
        "person_id": gallery_person_ids[best_index],
    }


# ==================================================
# Camera Thread
# ==================================================

def camera_reader():

    cap = cv2.VideoCapture(RTSP_URL)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Cannot connect to camera")
        stop_event.set()
        return

    print("Camera connected")

    try:
        while not stop_event.is_set():

            ret, frame = cap.read()

            if not ret:
                time.sleep(0.05)
                continue

            # Keep only the newest frame.
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass

            try:
                frame_queue.put_nowait(frame)
            except:
                pass

    finally:
        cap.release()
        print("Camera thread stopped")


# ==================================================
# AI Thread
# ==================================================

def ai_worker():

    print("AI thread started")

    process = psutil.Process()
    process.cpu_percent(interval=None)

    frame_count = 0
    tracked_faces = []
    last_entry_times = {}

    previous_time = time.time()

    # Each tracked face stores its tracker, recognition,
    # current bounding box and last emotion result.

    try:
        db.connect(reuse_if_open=True)

        while not stop_event.is_set():

            try:
                frame = frame_queue.get(timeout=0.1)
            except Empty:
                continue

            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                print("AI worker: Invalid frame received; skipping.")
                continue
            
            frame_count += 1

            cpu = process.cpu_percent(interval=None)

            small_frame = cv2.resize(
                frame,
                None,
                fx=SCALE,
                fy=SCALE,
            )

            # ==================================================
            # Detection and Tracking
            # ==================================================

            if frame_count % DETECTION_INTERVAL == 0:

                # InsightFace returns both bounding boxes
                # and 512D embeddings.
                detected_faces = FACE_APP.get(small_frame)

                tracked_faces = []

                for detected in detected_faces:

                    x1, y1, x2, y2 = detected.bbox

                    left = max(
                        0, int(x1 / SCALE)
                    )
                    top = max(
                        0, int(y1 / SCALE)
                    )
                    right = min(
                        frame.shape[1],
                        int(x2 / SCALE),
                    )
                    bottom = min(
                        frame.shape[0],
                        int(y2 / SCALE),
                    )

                    if right <= left or bottom <= top:
                        continue

                    # Recognition is evaluated at the configured
                    # interval; detection already provides embeddings.
                    if frame_count % RECOGNITION_INTERVAL == 0:
                        result = recognize_face(detected.embedding)
                    else:
                        result = {
                            "name": "Unknown",
                            "distance": None,
                            "person_id": None,
                        }

                    tracker = cv2.legacy.TrackerKCF_create()

                    tracker.init(
                        frame,
                        (
                            left,
                            top,
                            right - left,
                            bottom - top,
                        ),
                    )

                    tracked_faces.append({
                        "tracker": tracker,
                        "box": (top, right, bottom, left),
                        "result": result,
                        "emotion": "unknown",
                        "emotion_score": 0.0,
                    })

            else:

                # Update existing KCF trackers.
                active_faces = []

                for face_state in tracked_faces:

                    success, box = face_state["tracker"].update(
                        frame
                    )

                    if not success:
                        continue

                    x, y, w, h = [
                        int(value) for value in box
                    ]

                    top = max(0, y)
                    left = max(0, x)

                    bottom = min(
                        frame.shape[0],
                        y + h,
                    )
                    right = min(
                        frame.shape[1],
                        x + w,
                    )

                    if right <= left or bottom <= top:
                        continue

                    face_state["box"] = (
                        top,
                        right,
                        bottom,
                        left,
                    )

                    active_faces.append(face_state)

                tracked_faces = active_faces

            # ==================================================
            # Emotion Detection and Entry Logging
            # ==================================================

            for face_state in tracked_faces:

                result = face_state["result"]
                best_name = result["name"]

                if best_name == "Unknown":
                    face_state["emotion"] = "unknown"
                    face_state["emotion_score"] = 0.0
                    continue

                top, right, bottom, left = face_state["box"]

                if frame_count % EMOTION_INTERVAL == 0:

                    face_crop = frame[
                        max(0, top):min(frame.shape[0], bottom),
                        max(0, left):min(frame.shape[1], right),
                    ]

                    if face_crop.size > 0:

                        emotion, emotion_score = detect_emotion(
                            face_crop
                        )

                        face_state["emotion"] = emotion
                        face_state["emotion_score"] = float(
                            emotion_score
                        )

                emotion = face_state["emotion"]
               

                
                
                # Log each recognized smiling entry.
                if emotion == "smile":
                    now = time.time()

                    person_id = result["person_id"]
                    last_entry = last_entry_times.get(person_id, 0)

                    if now - last_entry >= ENTRY_COOLDOWN:
                        current_datetime = datetime.now()
                        current_date = current_datetime.date()
                        current_time = current_datetime.replace(
                            microsecond=0
                        ).time()

                        try:
                            with db.atomic():
                                # Always find the earliest entry of today.
                                first_daily_log = (
                                    EntryLog
                                    .select()
                                    .where(
                                        (EntryLog.person == person_id)
                                        & (EntryLog.entry_date == current_date)
                                    )
                                    .order_by(
                                        EntryLog.entry_time.asc(),
                                        EntryLog.id.asc(),
                                    )
                                    .first()
                                )

                                if first_daily_log is None:
                                    # First entry of the day.
                                    EntryLog.create(
                                        person_id=person_id,
                                        entry_date=current_date,
                                        entry_time=current_time,
                                        exit_time=None,
                                    )

                                    print(
                                        f"First entry recorded: {best_name} "
                                        f"- {current_time}",
                                        flush=True,
                                    )

                                else:
                                    # Update exit time on the first entry only.
                                    (
                                        EntryLog
                                        .update(exit_time=current_time)
                                        .where(
                                            EntryLog.id == first_daily_log.id
                                        )
                                        .execute()
                                    )

                                    # Preserve this additional entry separately.
                                    EntryLog.create(
                                        person_id=person_id,
                                        entry_date=current_date,
                                        entry_time=current_time,
                                        exit_time=None,
                                    )

                                    print(
                                        f"Additional entry recorded: {best_name} "
                                        f"- {current_time}",
                                        flush=True,
                                    )

                            last_entry_times[person_id] = now

                        except Exception as error:
                            print(f"Entry log error: {error}", flush=True)

            # ==================================================
            # Prepare Results for Display
            # ==================================================

            result_data = []

            for face_state in tracked_faces:

                result = face_state["result"]

                result_data.append({
                    "box": face_state["box"],
                    "name": result["name"],
                    "distance": result["distance"],
                    "emotion": face_state["emotion"],
                    "emotion_score": face_state["emotion_score"],
                })

            current_time = time.time()

            fps = 1.0 / max(
                current_time - previous_time,
                0.001,
            )

            previous_time = current_time

            output = (
                frame,
                result_data,
                cpu,
                fps,
            )

            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except Empty:
                    pass

            try:
                result_queue.put_nowait(output)
            except:
                pass

    except Exception as error:
        print(f"AI worker error: {error}", flush=True)
        traceback.print_exc()

    finally:
        if not db.is_closed():
            db.close()

        close_emotion_detector()
        print("AI thread stopped")


# ==================================================
# Main Display Loop
# ==================================================

def main():

    print("MAIN STARTED")

    camera_thread = Thread(
        target=camera_reader,
        daemon=True,
    )

    ai_thread = Thread(
        target=ai_worker,
        daemon=True,
    )

    camera_thread.start()
    ai_thread.start()

    print("Camera thread started")
    print("AI thread started")

    try:
        while not stop_event.is_set():

            try:
                frame, results, cpu, fps = result_queue.get(
                    timeout=0.1
                )
            except Empty:
                continue

            for result in results:

                top, right, bottom, left = result["box"]

                best_name = result["name"]
                emotion = result["emotion"]

                if best_name == "Unknown":
                    message = "Unknown"
                    color = (0, 0, 255)

                elif emotion == "smile":
                    message = f"Hi {best_name} - Welcome!"
                    color = (0, 255, 0)

                else:
                    message = f"Hi {best_name} - Smile Please!"
                    color = (0, 255, 255)

                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    color,
                    2,
                )

                cv2.putText(
                    frame,
                    message,
                    (left, max(30, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            cv2.putText(
                frame,
                f"CPU: {cpu:.1f}%",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Q pressed")
                stop_event.set()
                break

    finally:
        stop_event.set()

        camera_thread.join(timeout=2)
        ai_thread.join(timeout=2)

        cv2.destroyAllWindows()

        print("Program stopped")


if __name__ == "__main__":
    main()
