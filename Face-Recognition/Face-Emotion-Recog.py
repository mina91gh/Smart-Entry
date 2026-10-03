
import time

import cv2
import face_recognition
import numpy as np

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


previous_time = time.time()


# ==================================================
# Main Loop
# ==================================================

while True:

    # ------------------------------------------------
    # Read frame
    # ------------------------------------------------

    ret, frame = cap.read()


    if not ret:

        print("ERROR: Cannot read frame")
        break


    # ------------------------------------------------
    # Convert BGR → RGB
    # face_recognition needs RGB
    # ------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ------------------------------------------------
    # Detect faces
    # ------------------------------------------------

    face_locations = face_recognition.face_locations(
        rgb
    )


    # ------------------------------------------------
    # Calculate face encodings
    # ------------------------------------------------

    face_encodings = face_recognition.face_encodings(
        rgb,
        face_locations
    )


    # ==================================================
    # Process each face
    # ==================================================

    for (
        (top, right, bottom, left),
        encoding
    ) in zip(
        face_locations,
        face_encodings
    ):

        # ==============================================
        # Face Recognition
        # ==============================================

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

                best_distance = distances[
                    best_index
                ]


        # ==============================================
        # Emotion Detection
        # ==============================================

        emotion = "unknown"
        emotion_score = 0.0


        # فقط برای افراد شناخته‌شده Smile را بررسی می‌کنیم
        if best_name != "Unknown":

            # Make sure face coordinates
            # are inside the frame

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


            # Crop the detected face

            face_crop = frame[
                top_crop:bottom_crop,
                left_crop:right_crop
            ]


            if face_crop.size > 0:

                emotion, emotion_score = detect_emotion(
                    face_crop
                )


        # ==============================================
        # Access Message
        # ==============================================

        if best_name == "Unknown":

            message = "Unknown"

        elif emotion == "smile":

            message = f"Hi {best_name} - Welcome!"

        else:

            message = f"Hi {best_name} - Smile Please!"


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
        # Draw Message Above Face Box
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
    # FPS
    # ==================================================

    current_time = time.time()


    fps = 1 / (
        current_time -
        previous_time
    )


    previous_time = current_time

    '''
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    '''

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
