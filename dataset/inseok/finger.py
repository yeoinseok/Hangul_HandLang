from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ===== Settings =====
INPUT_ROOT = Path(r"C:\Users\kccistc\Desktop\DataSet")
OUTPUT_ROOT = Path(r"C:\Users\kccistc\Desktop\DataSet_cropped")
DEBUG_ROOT = Path(r"C:\Users\kccistc\Desktop\DataSet_crop_debug")
MODEL_PATH = Path(r"C:\Users\kccistc\Desktop\DataSet\hand_landmarker.task")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CROP_MARGIN = 80
SAVE_SIZE = 224
DRAW_DEBUG = True
REMOVE_BACKGROUND = True
MIN_DETECTION_CONFIDENCE = 0.2
# ====================


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def get_square_hand_crop(image, hand_landmarks):
    h, w, _ = image.shape

    xs = [int(lm.x * w) for lm in hand_landmarks]
    ys = [int(lm.y * h) for lm in hand_landmarks]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    hand_w = x_max - x_min
    hand_h = y_max - y_min
    box_size = max(hand_w, hand_h) + CROP_MARGIN * 2

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    x1 = max(0, cx - box_size // 2)
    y1 = max(0, cy - box_size // 2)
    x2 = min(w, cx + box_size // 2)
    y2 = min(h, cy + box_size // 2)

    crop = image[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def remove_background_grabcut(crop):
    h, w = crop.shape[:2]

    if h < 10 or w < 10:
        return cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)

    mask = np.zeros((h, w), np.uint8)

    rect_margin = max(5, min(h, w) // 20)
    rect = (
        rect_margin,
        rect_margin,
        max(1, w - rect_margin * 2),
        max(1, h - rect_margin * 2),
    )

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(crop, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)

    alpha = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)

    result = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = alpha
    return result


def draw_debug_landmarks(image, hand_landmarks):
    h, w = image.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(image, points[start], points[end], (255, 0, 0), 2)

    for x, y in points:
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)


def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_ROOT}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )

    detector = vision.HandLandmarker.create_from_options(options)

    total = 0
    saved = 0
    failed = 0

    for class_dir in sorted(INPUT_ROOT.iterdir(), key=lambda p: p.name):
        if not class_dir.is_dir():
            continue

        if class_dir.name in {"handenv", "__pycache__"}:
            continue

        image_files = [p for p in class_dir.iterdir() if is_image(p)]
        image_files.sort(key=lambda p: p.name)

        print(f"\n[{class_dir.name}] {len(image_files)} images")

        for image_path in image_files:
            total += 1
            image = read_image(image_path)

            if image is None:
                print(f"READ_FAIL: {image_path}")
                failed += 1
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect(mp_image)

            if not results.hand_landmarks:
                print(f"NO_HAND: {image_path}")
                failed += 1
                continue

            hand_landmarks = results.hand_landmarks[0]
            crop, box = get_square_hand_crop(image, hand_landmarks)

            if crop.size == 0:
                print(f"CROP_FAIL: {image_path}")
                failed += 1
                continue

            if REMOVE_BACKGROUND:
                crop = remove_background_grabcut(crop)
                crop = cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
            else:
                crop = cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)

            relative_path = image_path.relative_to(INPUT_ROOT)

            if REMOVE_BACKGROUND:
                save_path = (OUTPUT_ROOT / relative_path).with_suffix(".png")
            else:
                save_path = OUTPUT_ROOT / relative_path

            if write_image(save_path, crop):
                saved += 1
            else:
                print(f"WRITE_FAIL: {save_path}")
                failed += 1
                continue

            if DRAW_DEBUG:
                x1, y1, x2, y2 = box
                debug = image.copy()
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_debug_landmarks(debug, hand_landmarks)
                write_image(DEBUG_ROOT / relative_path, debug)

    detector.close()

    print("\nDone")
    print(f"Total : {total}")
    print(f"Saved : {saved}")
    print(f"Failed: {failed}")
    print(f"Output: {OUTPUT_ROOT}")
    if DRAW_DEBUG:
        print(f"Debug : {DEBUG_ROOT}")


if __name__ == "__main__":
    main()