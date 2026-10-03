import cv2
import mediapipe as mp
import numpy as np


# =========================
# MediaPipe Face Landmarker
# =========================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================
# Smile Detection
# =========================
'''
def is_smiling(face_landmarks):

    # گوشه‌های دهان
    left_mouth = face_landmarks.landmark[61]
    right_mouth = face_landmarks.landmark[291]

    # بالا و پایین لب
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]

    # محاسبه عرض دهان
    mouth_width = np.sqrt(
        (right_mouth.x - left_mouth.x) ** 2 +
        (right_mouth.y - left_mouth.y) ** 2
    )

    # محاسبه ارتفاع دهان
    mouth_height = np.sqrt(
        (lower_lip.x - upper_lip.x) ** 2 +
        (lower_lip.y - upper_lip.y) ** 2
    )

    if mouth_width == 0:
        return False, 0

    ratio = mouth_height / mouth_width

    # گوشه‌های دهان
    mouth_center_y = (
        left_mouth.y + right_mouth.y
    ) / 2

    upper_lip_y = upper_lip.y

    # مقدار باز شدن دهان
    openness = mouth_height / mouth_width

    # Threshold اولیه
    #smiling = openness > 0.30
    smiling = False

    return smiling, openness

'''
def is_smiling(face_landmarks):

    # -------------------------
    # Landmark های دهان
    # -------------------------

    left_corner = face_landmarks.landmark[61]
    right_corner = face_landmarks.landmark[291]

    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]

    # -------------------------
    # عرض دهان
    # -------------------------

    mouth_width = np.sqrt(
        (right_corner.x - left_corner.x) ** 2 +
        (right_corner.y - left_corner.y) ** 2
    )

    # -------------------------
    # بازشدگی دهان
    # -------------------------

    mouth_height = np.sqrt(
        (lower_lip.x - upper_lip.x) ** 2 +
        (lower_lip.y - upper_lip.y) ** 2
    )

    if mouth_width == 0:
        return False, 0

    openness = mouth_height / mouth_width

    # -------------------------
    # موقعیت مرکز دهان
    # -------------------------

    mouth_center_y = (
        left_corner.y + right_corner.y
    ) / 2

    # -------------------------
    # بررسی بالا رفتن گوشه های لب
    # -------------------------

    left_lift = mouth_center_y - left_corner.y
    right_lift = mouth_center_y - right_corner.y

    corner_lift = (
        left_lift + right_lift
    ) / 2

    # -------------------------
    # Smile Score
    # -------------------------

    smile_score = (
        corner_lift * 3
        + openness * 0.5
    )

    # -------------------------
    # Threshold
    # -------------------------

    smiling = smile_score > 0.01

    return smiling, smile_score
# =========================
# Camera
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()


# =========================
# Main Loop
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read frame")
        break


    # BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Face Mesh
    results = face_mesh.process(rgb_frame)


    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            smiling, score = is_smiling(
                face_landmarks
            )


            # =========================
            # Message
            # =========================

            if smiling:

                message = "Welcome! :)"
                color = (0, 255, 0)

            else:

                message = "Please Smile :)"
                color = (0, 0, 255)


            # =========================
            # Get Face Position
            # =========================

            h, w, _ = frame.shape

            xs = [
                int(point.x * w)
                for point in face_landmarks.landmark
            ]

            ys = [
                int(point.y * h)
                for point in face_landmarks.landmark
            ]

            x1 = max(0, min(xs))
            y1 = max(0, min(ys))
            x2 = min(w, max(xs))
            y2 = min(h, max(ys))


            # =========================
            # Draw Face Box
            # =========================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )


            # =========================
            # Draw Message
            # =========================

            cv2.putText(
                frame,
                message,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )


            # =========================
            # Show Score
            # =========================

            cv2.putText(
                frame,
                f"Score: {score:.2f}",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


    # =========================
    # Display
    # =========================

    cv2.imshow(
        "Smile Detection",
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()