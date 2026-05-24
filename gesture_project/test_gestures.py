import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

LABEL_HR = {
    "none": "nema geste",
    "ok": "OK",
    "thumbs_up": "palac gore",
    "thumbs_down": "palac dolje",
    "peace": "mir (peace)",
    "pointing": "pokazivanje prstom",
    "love": "ljubav (I love you)",
    "rock": "rock",
    "mobitel": "mobitel (palac i mali prst)",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXPORT_DIR = os.path.join(PROJECT_ROOT, "exported_model")
CAMERA_ID = 0


def get_mp_hands():
    if hasattr(mp, "solutions"):
        return mp.solutions.hands, mp.solutions.drawing_utils, mp.solutions.drawing_styles
    raise RuntimeError(
        "Ova verzija MediaPipe-a nema mp.solutions. Instaliraj kompatibilnu verziju mediapipe==0.10.14."
    )


mp_hands, mp_drawing, mp_drawing_styles = get_mp_hands()


# ---------------------------------------------------------------------------
# Normalizacija 
# ---------------------------------------------------------------------------

def normalize_landmarks(vec: np.ndarray) -> np.ndarray:
    pts = vec.reshape(21, 3)
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[9])
    if scale > 1e-6:
        pts = pts / scale
    return pts.flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Učitavanje modela
# ---------------------------------------------------------------------------

def load_model(export_dir: str):
    try:
        import joblib
    except ImportError:
        print("Nedostaje joblib. Instaliraj: pip install joblib", file=sys.stderr)
        sys.exit(1)

    model_path = os.path.join(export_dir, "gesture_classifier.pkl")
    labels_path = os.path.join(export_dir, "gesture_labels.txt")

    if not os.path.isfile(model_path):
        print(f"Model ne postoji: {model_path}\nPrvo pokreni train_gestures.py.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(labels_path):
        print(f"Labele ne postoje: {labels_path}\nPrvo pokreni train_gestures.py.", file=sys.stderr)
        sys.exit(1)

    clf = joblib.load(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        label_names = [line.strip() for line in f if line.strip()]
    return clf, label_names


# ---------------------------------------------------------------------------
# Extract i predikcija
# ---------------------------------------------------------------------------

def extract_landmarks(image_bgr: np.ndarray, hands) -> tuple[np.ndarray | None, str | None]:
    """Izvuci i normaliziraj landmark vektor iz slike."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None, None
    lm = result.multi_hand_landmarks[0].landmark
    vec = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
    vec = normalize_landmarks(vec)
    handedness = None
    if result.multi_handedness:
        handedness = result.multi_handedness[0].classification[0].label
    return vec, handedness


def predict(clf, label_names: list[str], vec: np.ndarray) -> tuple[str, float]:
    proba = clf.predict_proba([vec])[0]
    idx = int(np.argmax(proba))
    name = label_names[idx] if idx < len(label_names) else str(idx)
    return name, float(proba[idx])


def draw_landmarks_on_frame(frame: np.ndarray, hands_result) -> None:
    if hands_result.multi_hand_landmarks:
        for hand_lm in hands_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )


# ---------------------------------------------------------------------------
# Testiranje na slici
# ---------------------------------------------------------------------------

def run_image(clf, label_names: list[str], image_path: str) -> None:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ne mogu otvoriti sliku: {image_path}", file=sys.stderr)
        sys.exit(1)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        draw_landmarks_on_frame(img, result)
        vec, hand = extract_landmarks(img, hands)

    if vec is None:
        print("Nema detektirane ruke u slici.")
        return

    name, score = predict(clf, label_names, vec)
    hr = LABEL_HR.get(name, name)
    print(f"Gesta: {hr} ({name}), pouzdanost: {score:.2f}, ruka: {hand or '?'}")

    cv2.putText(img, f"{hr} ({score:.2f}) [{hand or '?'}]", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Gesture test - slika", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Testiranje webcamom
# ---------------------------------------------------------------------------

def run_webcam(clf, label_names: list[str], camera_id: int) -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Ne mogu otvoriti kameru {camera_id}.", file=sys.stderr)
        sys.exit(1)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        print("Izlaz: tipka Q ili ESC.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)

            draw_landmarks_on_frame(frame, result)

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark
                vec = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
                vec = normalize_landmarks(vec)
                hand = result.multi_handedness[0].classification[0].label if result.multi_handedness else "?"
                name, score = predict(clf, label_names, vec)
                hr = LABEL_HR.get(name, name)
                line = f"{hr} ({score:.2f}) [{hand}]"
            else:
                line = "Nema geste"

            cv2.putText(frame, line, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Gesture test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Učitavanje modela iz: {EXPORT_DIR}")
    clf, label_names = load_model(os.path.abspath(EXPORT_DIR))
    run_webcam(clf, label_names, CAMERA_ID)


if __name__ == "__main__":
    main()