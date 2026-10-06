import time

import cv2
import face_recognition
import numpy as np
import psutil

from model import FaceEmbedding
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
DETECTION_INTERVAL = 8

# Run face recognition every N frames
RECOGNITION_INTERVAL = 10

# Run emotion detection every N frames
EMOTION_INTERVAL = 10

# Resize scale for AI processing
SCALE = 0.5

# RTSP camera
RTSP_URL = "rtsp://root:root@192.168.10.176:554/axis-media/media.amp"


# ==================================================
# Load known faces from database
# ==================================================

known_embeddings = []
known_names = []


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


print(
    f"Loaded {len(known_embeddings)} "
    f"face embeddings from database"
)


# ==================================================
# Camera
# ==================================================

cap = cv2.VideoCapture(RTSP_URL)


if not cap.isOpened():

    print("ERROR: Cannot connect to camera")
    exit()


# ==================================================
# CPU
# ==================================================

process = psutil.Process()

process.cpu_percent(interval=None)


# ==================================================
# Variables
# ==================================================

frame_count = 0

face_locations = []

face_encodings = []

face_results = []

emotion_results = {}

trackers = []


previous_time = time.time()


# ==================================================
# Main Loop
# ==================================================

while True:

    # ------------------------------------------------
    # CPU usage
    # ------------------------------------------------

    cpu = process.cpu_percent(interval=None)


    # ------------------------------------------------
    # Read frame
    # ------------------------------------------------

    ret, frame = cap.read()


    if not ret:

        print("ERROR: Cannot read frame")
        break


    frame_count += 1


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


            trackers.append(tracker)


    else:

        # ----------------------------------------------
        # Update trackers
        # ----------------------------------------------

        new_face_locations = []


        for tracker in trackers:

            success, box = tracker.update(frame)


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


                if right > left and bottom > top:

                    new_face_locations.append(
                        (
                            top,
                            right,
                            bottom,
                            left
                        )
                    )


        face_locations = new_face_locations


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

        face_encodings = face_recognition.face_encodings(
            small_rgb,
            small_locations_for_encoding
        )


        # ----------------------------------------------
        # Recognition Results
        # ----------------------------------------------

        face_results = []


        for encoding in face_encodings:

            best_name = "Unknown"
            best_distance = None


            if len(known_embeddings) > 0:

                distances = face_recognition.face_distance(
                    known_embeddings,
                    encoding
                )


                best_index = int(
                    np.argmin(distances)
                )


                if distances[best_index] <= TOLERANCE:

                    best_name = known_names[
                        best_index
                    ]

                    best_distance = float(
                        distances[best_index]
                    )


            face_results.append({
                "name": best_name,
                "distance": best_distance
            })


    # ==================================================
    # Process Each Face
    # ==================================================

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


                    emotion_results[face_index] = (
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
            color = (0, 0, 255)

        elif emotion == "smile":

            # Green
            color = (0, 255, 0)

        else:

            # Yellow
            color = (0, 255, 255)


        # ==============================================
        # Draw Face Box
        # ==============================================

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
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
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
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

        break


# ==================================================
# Cleanup
# ==================================================

cap.release()

cv2.destroyAllWindows()

close_emotion_detector()