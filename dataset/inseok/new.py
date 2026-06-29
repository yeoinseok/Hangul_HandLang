from pathlib import Path

import cv2
import numpy as np


# ===== Settings =====
INPUT_ROOT = Path(r"C:\Users\kccistc\Desktop\DataSet")
OUTPUT_ROOT = Path(r"C:\Users\kccistc\Desktop\DataSet_bg_removed")
DEBUG_ROOT = Path(r"C:\Users\kccistc\Desktop\DataSet_bg_debug")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SAVE_SIZE = 224
DRAW_DEBUG = True
RECT_MARGIN_RATIO = 0.08
# ====================


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


def remove_background(image):
    h, w = image.shape[:2]

    margin = int(min(h, w) * RECT_MARGIN_RATIO)
    margin = max(5, margin)

    rect = (
        margin,
        margin,
        max(1, w - margin * 2),
        max(1, h - margin * 2),
    )

    mask = np.zeros((h, w), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image, mask, rect, bg_model, fg_model, 8, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = 255
        return result, np.full((h, w), 255, dtype=np.uint8)

    alpha = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)

    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = alpha

    return result, alpha


def crop_to_alpha(image_bgra):
    alpha = image_bgra[:, :, 3]
    ys, xs = np.where(alpha > 10)

    if len(xs) == 0 or len(ys) == 0:
        return image_bgra

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    margin = 10
    h, w = alpha.shape

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    return image_bgra[y1:y2, x1:x2]


def square_resize_with_padding(image_bgra, size):
    h, w = image_bgra.shape[:2]

    scale = size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(image_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def make_debug(image, alpha):
    debug = image.copy()
    green = np.zeros_like(debug)
    green[:, :] = (0, 255, 0)

    mask = alpha > 10
    debug[mask] = cv2.addWeighted(debug[mask], 0.5, green[mask], 0.5, 0)

    return debug


def main():
    total = 0
    saved = 0
    failed = 0

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_ROOT}")

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

            removed, alpha = remove_background(image)
            cropped = crop_to_alpha(removed)
            result = square_resize_with_padding(cropped, SAVE_SIZE)

            relative_path = image_path.relative_to(INPUT_ROOT)
            save_path = (OUTPUT_ROOT / relative_path).with_suffix(".png")

            if write_image(save_path, result):
                saved += 1
            else:
                print(f"WRITE_FAIL: {save_path}")
                failed += 1
                continue

            if DRAW_DEBUG:
                debug = make_debug(image, alpha)
                write_image(DEBUG_ROOT / relative_path, debug)

    print("\nDone")
    print(f"Total : {total}")
    print(f"Saved : {saved}")
    print(f"Failed: {failed}")
    print(f"Output: {OUTPUT_ROOT}")
    if DRAW_DEBUG:
        print(f"Debug : {DEBUG_ROOT}")


if __name__ == "__main__":
    main()