import time

import cv2
import face_recognition
import numpy as np
import psutil

from threading import Thread, Event
from queue import Queue, Empty

from datetime import datetime

from model import FaceEmbedding, EntryLog
from Emotion_Detection import (
    detect_emotion,
    close_emotion_detector
)


# ==================================================
# Settings
# ==================================================

WINDOW_NAME = "Smart Entry System"

# Face recognition tolerance
# Lower value = stricter recognition
TOLERANCE = 0.5

# Run face detection every N frames
DETECTION_INTERVAL = 5

# Run face recognition every N frames
RECOGNITION_INTERVAL = 5

# Run emotion detection every N frames
EMOTION_INTERVAL = 5

# Resize scale for AI processing
SCALE = 0.5

ENTRY_COOLDOWN = 30

# RTSP camera
RTSP_URL = "rtsp://root:root@192.168.10.176:554/axis-media/media.amp"


# ==================================================
# Queues and Stop Event
# ==================================================

frame_queue = Queue(maxsize=1)

result_queue = Queue(maxsize=1)

stop_event = Event()


# ==================================================
# Load known faces from database
# ==================================================

known_embeddings = []
known_names = []

known_person_ids = []
last_entry_times = {}

for emb in FaceEmbedding.select():

    known_embeddings.append(
        np.array(
            emb.embedding,
            dtype=np.float64
        )
    )

    known_names.append(
        emb.person.name
    )
    known_person_ids.append(
        emb.person.id
    )
print(
    f"Loaded {len(known_embeddings)} "
    f"face embeddings from database"
)


# ==================================================
# Camera Thread
# ==================================================

def camera_reader():

    cap = cv2.VideoCapture(RTSP_URL)

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if not cap.isOpened():

        print("ERROR: Cannot connect to camera")

        stop_event.set()

        return

    print("Camera connected")

    while not stop_event.is_set():

        ret, frame = cap.read()

        if not ret:

            print("ERROR: Cannot read frame")

            continue

        try:

            # Remove old frame
            if frame_queue.full():

                frame_queue.get_nowait()

            # Add newest frame
            frame_queue.put_nowait(frame)

        except:

            pass

    cap.release()

    print("Camera thread stopped")


# ==================================================
# AI Thread
# ==================================================

def ai_worker():

    print("AI thread started")

    process = psutil.Process()

    process.cpu_percent(
        interval=None
    )

    frame_count = 0

    face_locations = []

    face_encodings = []

    face_results = []

    emotion_results = {}

    trackers = []

    previous_time = time.time()

    while not stop_event.is_set():

        # ------------------------------------------------
        # Get newest frame
        # ------------------------------------------------

        try:

            frame = frame_queue.get(
                timeout=0.1
            )

        except Empty:

            continue


        frame_count += 1


        # ------------------------------------------------
        # CPU
        # ------------------------------------------------

        cpu = process.cpu_percent(
            interval=None
        )


        # ==================================================
        # Resize frame for AI processing
        # ==================================================

        small_frame = cv2.resize(
            frame,
            None,
            fx=SCALE,
            fy=SCALE
        )


        small_rgb = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )


        # ==================================================
        # Face Detection + Tracker
        # ==================================================

        if frame_count % DETECTION_INTERVAL == 0:

            # ----------------------------------------------
            # Face Detection
            # ----------------------------------------------

            small_face_locations = (
                face_recognition.face_locations(
                    small_rgb
                )
            )


            # ----------------------------------------------
            # Convert coordinates
            # small image → original frame
            # ----------------------------------------------

            face_locations = [

                (
                    int(top / SCALE),
                    int(right / SCALE),
                    int(bottom / SCALE),
                    int(left / SCALE)
                )

                for top, right, bottom, left
                in small_face_locations
            ]


            # ----------------------------------------------
            # Create new trackers
            # ----------------------------------------------

            trackers = []


            for top, right, bottom, left in face_locations:

                width = right - left

                height = bottom - top


                if width <= 0 or height <= 0:

                    continue


                tracker = cv2.TrackerKCF_create()


                tracker.init(
                    frame,
                    (
                        left,
                        top,
                        width,
                        height
                    )
                )


                trackers.append(
                    tracker
                )


        else:

            # ----------------------------------------------
            # Update trackers
            # ----------------------------------------------

            new_face_locations = []


            for tracker in trackers:

                success, box = tracker.update(
                    frame
                )


                if success:

                    x, y, w, h = [

                        int(value)

                        for value in box
                    ]


                    top = max(
                        0,
                        y
                    )

                    left = max(
                        0,
                        x
                    )

                    bottom = min(
                        frame.shape[0],
                        y + h
                    )

                    right = min(
                        frame.shape[1],
                        x + w
                    )


                    if (
                        right > left
                        and bottom > top
                    ):

                        new_face_locations.append(
                            (
                                top,
                                right,
                                bottom,
                                left
                            )
                        )


            face_locations = (
                new_face_locations
            )


        # ==================================================
        # Face Recognition
        # ==================================================

        if (
            frame_count % RECOGNITION_INTERVAL == 0
            and len(face_locations) > 0
        ):

            # ----------------------------------------------
            # Convert original coordinates
            # → small image coordinates
            # ----------------------------------------------

            small_locations_for_encoding = [

                (
                    int(top * SCALE),
                    int(right * SCALE),
                    int(bottom * SCALE),
                    int(left * SCALE)
                )

                for top, right, bottom, left
                in face_locations
            ]


            # ----------------------------------------------
            # Face Encoding
            # ----------------------------------------------

            face_encodings = (
                face_recognition.face_encodings(
                    small_rgb,
                    small_locations_for_encoding
                )
            )


            # ----------------------------------------------
            # Recognition Results
            # ----------------------------------------------

            face_results = []


            for encoding in face_encodings:

                best_name = "Unknown"

                best_distance = None

                best_person_id = None

                if len(known_embeddings) > 0:

                    distances = (
                        face_recognition.face_distance(
                            known_embeddings,
                            encoding
                        )
                    )


                    best_index = int(
                        np.argmin(distances)
                    )


                    if (
                        distances[best_index]
                        <= TOLERANCE
                    ):

                        best_name = (
                            known_names[
                                best_index
                            ]
                        )

                        best_distance = float(
                            distances[
                                best_index
                            ]
                        )

                        best_person_id = known_person_ids[best_index]


                face_results.append({

                    "name": best_name,

                    "distance": best_distance,

                    "person_id": best_person_id
                })


        # ==================================================
        # Process Each Face
        # ==================================================

        current_emotions = {}


        for face_index, (

            (top, right, bottom, left),

            result

        ) in enumerate(

            zip(
                face_locations,
                face_results
            )

        ):

            best_name = result["name"]

            best_distance = result["distance"]


            # ==============================================
            # Emotion Detection
            # ==============================================

            emotion = "unknown"

            emotion_score = 0.0


            # Only check smile for known people
            if best_name != "Unknown":

                if frame_count % EMOTION_INTERVAL == 0:

                    # Make sure coordinates are inside frame

                    top_crop = max(
                        0,
                        top
                    )

                    bottom_crop = min(
                        frame.shape[0],
                        bottom
                    )

                    left_crop = max(
                        0,
                        left
                    )

                    right_crop = min(
                        frame.shape[1],
                        right
                    )


                    # Crop face

                    face_crop = frame[
                        top_crop:bottom_crop,
                        left_crop:right_crop
                    ]


                    if face_crop.size > 0:

                        emotion, emotion_score = (
                            detect_emotion(
                                face_crop
                            )
                        )


                        emotion_results[
                            face_index
                        ] = (
                            emotion,
                            emotion_score
                        )


                else:

                    # Use previous emotion result

                    if face_index in emotion_results:

                        emotion, emotion_score = (
                            emotion_results[
                                face_index
                            ]
                        )


            current_emotions[
                face_index
            ] = (
                emotion,
                emotion_score
            )


            # ==================================================
            # Entry Logging
            # ==================================================

            if best_name != "Unknown" and emotion == "smile":

                now = time.time()

                last_entry = last_entry_times.get(
                    best_name,
                    0
                )

                if now - last_entry >= ENTRY_COOLDOWN:

                    current_datetime = datetime.now()

                    EntryLog.create(

                        person_id=result["person_id"],

                        entry_date=current_datetime.date(),

                        #entry_time=current_datetime.time()
                        entry_time=current_datetime.replace(
                            microsecond=0
                        ).time()
                    )

                    last_entry_times[best_name] = now

                    print(
                        f"Entry logged: "
                        f"{best_name} - "
                        f"{current_datetime}"
                    )

        # ==================================================
        # Prepare Result
        # ==================================================

        result_data = []

        for face_index, (

            (top, right, bottom, left),

            result

        ) in enumerate(

            zip(
                face_locations,
                face_results
            )

        ):

            emotion, emotion_score = (
                current_emotions.get(
                    face_index,
                    ("unknown", 0.0)
                )
            )


            result_data.append({

                "box": (
                    top,
                    right,
                    bottom,
                    left
                ),

                "name": result["name"],

                "distance": result["distance"],

                "emotion": emotion,

                "emotion_score": emotion_score

            })


        # ==================================================
        # FPS
        # ==================================================

        current_time = time.time()

        fps = 1 / max(
            current_time - previous_time,
            0.001
        )

        previous_time = current_time


        # ==================================================
        # Send Result to Main Thread
        # ==================================================

        try:

            if result_queue.full():

                result_queue.get_nowait()


            result_queue.put_nowait(
                (
                    frame,
                    result_data,
                    cpu,
                    fps
                )
            )

        except:

            pass


    # ==================================================
    # Cleanup AI Thread
    # ==================================================

    close_emotion_detector()

    print("AI thread stopped")


# ==================================================
# Main
# ==================================================

def main():

    print("MAIN STARTED")


    # ==================================================
    # Create Threads
    # ==================================================

    camera_thread = Thread(

        target=camera_reader,

        daemon=True
    )


    ai_thread = Thread(

        target=ai_worker,

        daemon=True
    )


    # ==================================================
    # Start Threads
    # ==================================================

    camera_thread.start()

    ai_thread.start()


    print("Camera thread started")

    print("AI thread started")


    # ==================================================
    # Main Display Loop
    # ==================================================

    try:

        while not stop_event.is_set():

            try:

                frame, results, cpu, fps = (
                    result_queue.get(
                        timeout=0.1
                    )
                )

            except Empty:

                continue


            # ==================================================
            # Draw Results
            # ==================================================

            for result in results:

                top, right, bottom, left = (
                    result["box"]
                )


                best_name = result["name"]

                emotion = result["emotion"]


                # ==============================================
                # Access Message
                # ==============================================

                if best_name == "Unknown":

                    message = "Unknown"


                elif emotion == "smile":

                    message = (
                        f"Hi {best_name} - Welcome!"
                    )


                else:

                    message = (
                        f"Hi {best_name} - Smile Please!"
                    )


                # ==============================================
                # Box Color
                # ==============================================

                if best_name == "Unknown":

                    # Red

                    color = (
                        0,
                        0,
                        255
                    )


                elif emotion == "smile":

                    # Green

                    color = (
                        0,
                        255,
                        0
                    )


                else:

                    # Yellow

                    color = (
                        0,
                        255,
                        255
                    )


                # ==============================================
                # Draw Face Box
                # ==============================================

                cv2.rectangle(

                    frame,

                    (
                        left,
                        top
                    ),

                    (
                        right,
                        bottom
                    ),

                    color,

                    2
                )


                # ==============================================
                # Draw Message
                # ==============================================

                cv2.putText(

                    frame,

                    message,

                    (
                        left,
                        max(
                            30,
                            top - 10
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    color,

                    2
                )


            # ==================================================
            # CPU Display
            # ==================================================

            cv2.putText(

                frame,

                f"CPU: {cpu:.1f}%",

                (
                    20,
                    30
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (
                    255,
                    255,
                    255
                ),

                2
            )


            # ==================================================
            # FPS Display
            # ==================================================

            cv2.putText(

                frame,

                f"FPS: {fps:.1f}",

                (
                    20,
                    60
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (
                    255,
                    255,
                    255
                ),

                2
            )


            # ==================================================
            # Display
            # ==================================================

            cv2.imshow(

                WINDOW_NAME,

                frame
            )


            # ==================================================
            # Keyboard
            # ==================================================

            key = cv2.waitKey(1) & 0xFF


            if key == ord("q"):

                print("Q pressed")

                stop_event.set()

                break


    finally:

        # ==================================================
        # Stop Threads
        # ==================================================

        stop_event.set()


        # Wait for Camera Thread

        camera_thread.join(
            timeout=2
        )


        # Wait for AI Thread

        ai_thread.join(
            timeout=2
        )


        cv2.destroyAllWindows()


        print("Program stopped")


# ==================================================
# Run
# ==================================================

if __name__ == "__main__":

    main()

