import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

GESTURE_FOLDERS = [
    "none",
    "ok",
    "thumbs_up",
    "thumbs_down",
    "peace",
    "pointing",
    "love",
    "rock",
    "mobitel",
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

KEY_TO_GESTURE = {
    ord("1"): "none",
    ord("2"): "ok",
    ord("3"): "thumbs_up",
    ord("4"): "thumbs_down",
    ord("5"): "peace",
    ord("6"): "pointing",
    ord("7"): "love",
    ord("8"): "rock",
    ord("9"): "mobitel",
}

LABEL_HR = {
    "none": "nema geste",
    "ok": "OK",
    "thumbs_up": "palac gore",
    "thumbs_down": "palac dolje",
    "peace": "mir",
    "pointing": "pokazivanje prstom",
    "love": "ljubav",
    "rock": "rock",
    "mobitel": "mobitel",
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = str(PROJECT_ROOT / "dataset")
EXPORT_DIR = str(PROJECT_ROOT / "exported_model")
N_ESTIMATORS = 300
MAX_DEPTH = 18
MIN_SAMPLES_LEAF = 3
VAL_FRACTION = 0.2
CAMERA_ID = 0
SAVE_INTERVAL = 0.3


def get_mp_hands():
    if hasattr(mp, "solutions"):
        return mp.solutions.hands, mp.solutions.drawing_utils, mp.solutions.drawing_styles
    raise RuntimeError(
        "Ova verzija MediaPipe-a nema mp.solutions. Instaliraj kompatibilnu verziju, npr. mediapipe==0.10.14."
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
# MediaPipe pomoćne funkcije
# ---------------------------------------------------------------------------

def ensure_dataset_root(data_dir: str) -> None:
    root = Path(data_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in GESTURE_FOLDERS:
        (root / name).mkdir(parents=True, exist_ok=True)


def validate_dataset_root(data_dir: str) -> None:
    root = os.path.abspath(data_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset mapa ne postoji: {root}")

    available = {name.lower(): name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))}
    missing = [f for f in GESTURE_FOLDERS if f.lower() not in available]
    if missing:
        raise ValueError("Nedostaju podmape: " + ", ".join(missing) + f"\nOčekivano u: {root}")

    for folder in GESTURE_FOLDERS:
        actual = available[folder.lower()]
        entries = os.listdir(os.path.join(root, actual))
        n_img = len([p for p in entries if p.lower().endswith(IMAGE_EXTENSIONS)])
        n_npy = len([p for p in entries if p.lower().endswith(".npy")])
        if n_img == 0 and n_npy == 0:
            raise ValueError(f"Mapa '{actual}' nema slikovnih ni .npy datoteka.")


def extract_landmarks(image_bgr: np.ndarray, hands) -> np.ndarray | None:
    """Izvuci i normaliziraj landmark vektor iz slike."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0].landmark
    vec = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
    return normalize_landmarks(vec)


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
# Prikupljanje uzoraka kamerom
# ---------------------------------------------------------------------------

def collect_samples(data_dir: str, camera_id: int, save_interval: float) -> None:
    root = Path(data_dir).resolve()
    ensure_dataset_root(str(root))

    sample_index = {g: len(list((root / g).glob("*.npy"))) for g in GESTURE_FOLDERS}
    active_gesture = "none"
    last_save_time = 0.0

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Ne mogu otvoriti kameru {camera_id}.", file=sys.stderr)
        sys.exit(1)

    print("Tipke: 1-9 odabir geste | s spremi | q izlaz")
    print("1 none, 2 ok, 3 thumbs_up, 4 thumbs_down, 5 peace, 6 pointing, 7 love, 8 rock, 9 mobitel")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
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
                hand_label = result.multi_handedness[0].classification[0].label if result.multi_handedness else "?"
                status = f"Gesta: {LABEL_HR.get(active_gesture, active_gesture)} | Kamera: {hand_label} | S za spremanje"
            else:
                status = f"Gesta: {LABEL_HR.get(active_gesture, active_gesture)} | Nema detektirane ruke | S za spremanje"

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Spremanja: {sample_index[active_gesture]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Prikupljanje gesta", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            if key in KEY_TO_GESTURE:
                active_gesture = KEY_TO_GESTURE[key]
                print(f"Odabrana gesta: {active_gesture}")

            if key in (ord("s"), ord("S")):
                now = cv2.getTickCount() / cv2.getTickFrequency()
                if now - last_save_time < save_interval:
                    continue
                if not result.multi_hand_landmarks:
                    print("Nema ruke za spremanje.")
                    continue

                vec = extract_landmarks(frame, hands)
                if vec is None:
                    print("Nije moguće izvući landmarke.")
                    continue

                folder = root / active_gesture
                folder.mkdir(parents=True, exist_ok=True)
                idx = sample_index[active_gesture]
                out_path = folder / f"{active_gesture}_{idx:05d}.npy"
                np.save(out_path, vec)
                sample_index[active_gesture] += 1
                last_save_time = now
                print(f"Spremljeno: {out_path}")

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Učitavanje dataseta
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    root = os.path.abspath(data_dir)

    X, y = [], []
    label_names = GESTURE_FOLDERS
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}

    total_images = 0
    skipped = 0

    for label in label_names:
        folder = Path(root) / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Mapa gesti nije pronađena: {folder}")
        npy_files = sorted(folder.glob("*.npy"))
        img_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

        if npy_files:
            print(f"[{label}] {len(npy_files)} uzoraka (.npy)")
            for f in npy_files:
                arr = np.load(f)
                if arr is None or arr.shape != (63,):
                    skipped += 1
                    continue
                X.append(arr.astype(np.float32))
                y.append(label_to_idx[label])
                total_images += 1
        else:
            print(f"[{label}] {len(img_files)} slika")
            with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3) as hands:
                for fname in img_files:
                    total_images += 1
                    img = cv2.imread(str(fname))
                    if img is None:
                        skipped += 1
                        continue
                    vec = extract_landmarks(img, hands)
                    if vec is None:
                        skipped += 1
                        continue
                    X.append(vec)
                    y.append(label_to_idx[label])

    print(f"\nUkupno uzoraka: {total_images}, uspješno: {len(X)}, preskočeno: {skipped}")
    if not X:
        raise RuntimeError("Nema spremljenih uzoraka za trening.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), label_names


# ---------------------------------------------------------------------------
# Trening i export
# ---------------------------------------------------------------------------

def train_and_export(data_dir: str, export_dir: str, n_estimators: int, val_fraction: float) -> None:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        import joblib
    except ImportError:
        print("Nedostaju ovisnosti. Instaliraj:\n  pip install scikit-learn joblib", file=sys.stderr)
        sys.exit(1)

    print("Učitavanje dataseta i ekstrakcija landmark-a ...")
    X, y, label_names = load_dataset(data_dir)

    if len(set(y.tolist())) < 2:
        raise RuntimeError("Trebaš barem dvije različite klase za trening.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=42, stratify=y
    )

    print(f"Trening: {len(X_train)}, validacija: {len(X_val)}")
    print(f"\nTreniranje RandomForest (n_estimators={n_estimators}, max_depth={MAX_DEPTH}, min_samples_leaf={MIN_SAMPLES_LEAF}) ...")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("\n── Validacijski izvještaj ──────────────────────────────")
    print(classification_report(y_val, y_pred, target_names=label_names, zero_division=0))

    os.makedirs(export_dir, exist_ok=True)
    model_path = os.path.join(export_dir, "gesture_classifier.pkl")
    labels_path = os.path.join(export_dir, "gesture_labels.txt")
    joblib.dump(clf, model_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_names))

    print(f"Model spremljen: {model_path}")
    print(f"Labele spremljene: {labels_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Pokreće se trening gesti s hardkodiranim parametrima.")
    print(f"Dataset: {DATA_DIR}, export: {EXPORT_DIR}, kamera: {CAMERA_ID}")
    ensure_dataset_root(DATA_DIR)
    collect_samples(DATA_DIR, CAMERA_ID, SAVE_INTERVAL)
    train_and_export(DATA_DIR, EXPORT_DIR, N_ESTIMATORS, VAL_FRACTION)


if __name__ == "__main__":
    main()