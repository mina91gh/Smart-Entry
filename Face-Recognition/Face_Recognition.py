import time

import cv2
import face_recognition
import numpy as np

from model import FaceEmbedding


# =========================
# Settings
# =========================

RTSP_URL = "rtsp://root:root@192.168.10.176:554/axis-media/media.amp"

WINDOW_NAME = "Smart Entry System"

# face_recognition distances: 0 = identical, ~0.6+ = different person
TOLERANCE = 0.5


# =========================
# Load known faces from the database
# =========================

known_embeddings = []
known_names = []

for emb in FaceEmbedding.select():

    known_embeddings.append(
        np.array(emb.embedding, dtype=np.float64)
    )
    known_names.append(emb.person.name)


print(f"Loaded {len(known_embeddings)} face embeddings from database")


# =========================
# Camera
# =========================

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("ERROR: Cannot connect to camera")
    exit()


previous_time = time.time()


# =========================
# Main Loop
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read frame")
        break


    # face_recognition needs RGB, OpenCV gives BGR
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # -------------------------
    # Detect faces + compute encodings
    # -------------------------

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)


    for (top, right, bottom, left), encoding in zip(
        face_locations,
        face_encodings
    ):

        # -------------------------
        # Find best match in the database
        # -------------------------

        best_name = "Unknown"
        best_distance = None

        if len(known_embeddings) > 0:

            distances = face_recognition.face_distance(
                known_embeddings,
                encoding
            )

            best_index = int(np.argmin(distances))

            if distances[best_index] <= TOLERANCE:
                best_name = known_names[best_index]
                best_distance = distances[best_index]


        # -------------------------
        # Draw
        # -------------------------

        color = (0, 255, 0) if best_name != "Unknown" else (0, 0, 255)

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            color,
            2
        )

        label = best_name if best_distance is None else (
            f"{best_name} ({best_distance:.2f})"
        )

        cv2.putText(
            frame,
            label,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


    # -------------------------
    # FPS
    # -------------------------

    current_time = time.time()

    fps = 1 / (current_time - previous_time)

    previous_time = current_time


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # -------------------------
    # Display
    # -------------------------

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()
