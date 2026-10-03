
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def detect_smile(face_landmarks):

    left_corner = face_landmarks.landmark[61]
    right_corner = face_landmarks.landmark[291]

    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]

    mouth_width = np.sqrt(
        (right_corner.x - left_corner.x) ** 2 +
        (right_corner.y - left_corner.y) ** 2
    )

    mouth_height = np.sqrt(
        (lower_lip.x - upper_lip.x) ** 2 +
        (lower_lip.y - upper_lip.y) ** 2
    )

    if mouth_width == 0:
        return False, 0.0

    openness = mouth_height / mouth_width

    mouth_center_y = (
        left_corner.y +
        right_corner.y
    ) / 2

    left_lift = mouth_center_y - left_corner.y
    right_lift = mouth_center_y - right_corner.y

    corner_lift = (left_lift + right_lift) / 2

    score = corner_lift * 3 + openness * 0.5

    smiling = score > 0.02

    return smiling, score


def detect_emotion(frame):

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return "unknown", 0.0

    face_landmarks = results.multi_face_landmarks[0]

    smiling, score = detect_smile(face_landmarks)

    if smiling:
        return "smile", score

    return "neutral", score


def close_emotion_detector():
    face_mesh.close()


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        exit()

    print("Camera started...")
    print("Press Q to quit")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Cannot read frame")
            break

        emotion, score = detect_emotion(frame)

        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Score: {score:.3f}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Emotion Detection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    close_emotion_detector()

