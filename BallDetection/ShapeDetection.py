import json
import argparse
import cv2
import numpy as np
from typing import List, Tuple


def shape_score(contour: np.ndarray, cfg: dict) -> float:
    """
    Compute an elliptical score for a contour.
    Score = minor_axis_length / major_axis_length (0.0 to 1.0).
    """
    if len(contour) < 5:
        return 0.0
    ellipse = cv2.fitEllipse(contour)
    (_, _), (major, minor), _ = ellipse
    if major == 0:
        return 0.0
    return min(major, minor) / max(major, minor)


def preprocess_mask(frame: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Generate binary mask for ping-pong ball based on thresholds.
    cfg keys:
      "color_space", "lower_thresh", "upper_thresh", "lab_thresh", "lab_ref"
    """
    lab_thresh = cfg.get('lab_thresh', None)
    if lab_thresh is not None:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab).astype(np.float32)
        ref = np.array(cfg.get('lab_ref', [200, 128, 128]), dtype=np.float32)
        dist = np.linalg.norm(lab - ref[None, None, :], axis=2)
        mask = (dist < lab_thresh).astype(np.uint8) * 255
    else:
        space = cfg.get('color_space', 'HSV')
        conv = cv2.COLOR_BGR2HSV if space == 'HSV' else cv2.COLOR_BGR2YCrCb
        cs = cv2.cvtColor(frame, conv)
        lower = np.array(cfg.get('lower_thresh', [0, 0, 200]), dtype=np.uint8)
        upper = np.array(cfg.get('upper_thresh', [180, 30, 255]), dtype=np.uint8)
        mask = cv2.inRange(cs, lower, upper)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    return mask


def Shape_Detection(frames: np.ndarray, frame_index: int, cfg: dict) -> List[Tuple[np.ndarray, int]]:
    """
    Detect contours matching ping-pong ball shape in a frame.
    cfg keys:
      "min_area", "max_area", "convex_thresh"
    Returns list of (contour, index).
    """
    frame = frames[frame_index]
    mask = preprocess_mask(frame, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    min_a = cfg.get('min_area', 50)
    max_a = cfg.get('max_area', 5000)
    conv_th = cfg.get('convex_thresh', 0.8)
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_a or area > max_a:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0 or (area / hull_area) < conv_th:
            continue
        results.append((cnt, i))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Standalone test for shape detection and ellipse scoring with pause control"
    )
    parser.add_argument('config', help='Path to JSON config file')
    parser.add_argument('--export-config', dest='export_config', help='Optional path to re-export full config')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        full_cfg = json.load(f)
    cfg = full_cfg['shape_detection']

    frames = np.load(cfg['video_npy'])
    window = cfg.get('shape_window', 'Shape Detection')
    delay = int(cfg.get('display_fps_delay', 30))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    idx = 0
    num = len(frames)
    paused = False

    while idx < num:
        frame = frames[idx]
        detections = Shape_Detection(frames, idx, cfg)
        display = frame.copy()
        for cnt, _ in detections:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            score = shape_score(cnt, cfg)
            cv2.putText(display, f"{score:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow(window, display)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar toggles pause  # Enter key toggles pause
            paused = not paused
        if not paused:
            idx += 1

    cv2.destroyAllWindows()
    if args.export_config:
        with open(args.export_config, 'w', encoding='utf-8') as ef:
            json.dump(full_cfg, ef, indent=4)
        print(f"Config exported to {args.export_config}")

if __name__ == '__main__':
    main()
