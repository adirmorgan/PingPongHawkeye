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
    roi_x_min, roi_x_max, roi_y_min, roi_y_max = full_cfg.get("roi_bounds", [320, 900, 200, 470])
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
        info_texts = []

        hull_best = (-1, -1)
        ellipse_best = (-1, -1)
        shape_best = (-1, -1)

        for i, (cnt, _) in enumerate(detections):
            x, y, w, h = cv2.boundingRect(cnt)
            if not (roi_x_min <= x <= roi_x_max and roi_y_min <= y <= roi_y_max):
                continue

            area = float(cv2.contourArea(cnt))
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            hull_ratio = area / hull_area if hull_area > 0 else 0.0

            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (_, _), (major, minor), _ = ellipse
                ellipse_area = np.pi * (major / 2) * (minor / 2)
                ellipse_ratio = area / ellipse_area if ellipse_area > 0 else 0.0
            else:
                ellipse_ratio = 0.0

            shape_ratio = shape_score(cnt, cfg)

            if hull_ratio > hull_best[1]:
                hull_best = (i, hull_ratio)
            if ellipse_ratio > ellipse_best[1]:
                ellipse_best = (i, ellipse_ratio)
            if shape_ratio > shape_best[1]:
                shape_best = (i, shape_ratio)

            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            contour_id = f"#{i + 1}"
            cv2.putText(display, contour_id, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            info_line = f"{contour_id} | Hull: {hull_ratio:.2f}, Ellipse: {ellipse_ratio:.2f}, Shape: {shape_ratio:.2f}"
            info_texts.append((i, info_line, (x, y, w, h)))

        best_vals = {'Hull': 0, 'Ellipse': 0, 'Shape': 0}
        best_indices = {'Hull': -1, 'Ellipse': -1, 'Shape': -1}
        for j, (i, text, _) in enumerate(info_texts):
            parts = text.split(',')
            hull_val = float(parts[0].split("Hull: ")[1])
            ellipse_val = float(parts[1].split("Ellipse: ")[1])
            shape_val = float(parts[2].split("Shape: ")[1])
            if hull_val > best_vals['Hull']:
                best_vals['Hull'] = hull_val
                best_indices['Hull'] = j
            if ellipse_val > best_vals['Ellipse']:
                best_vals['Ellipse'] = ellipse_val
                best_indices['Ellipse'] = j
            if shape_val > best_vals['Shape']:
                best_vals['Shape'] = shape_val
                best_indices['Shape'] = j

        for j, (_, text, _) in enumerate(info_texts):
            cy = 60 + j * 20
            x_text = 10
            cv2.putText(display, text, (x_text, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for key, idx_b in best_indices.items():
                if idx_b == j:
                    parts = text.split(',')
                    if key == 'Hull':
                        prefix = parts[0].split("Hull: ")[0] + "Hull: "
                        number = parts[0].split("Hull: ")[1]
                    elif key == 'Ellipse':
                        prefix = parts[0] + ',' + parts[1].split("Ellipse: ")[0] + "Ellipse: "
                        number = parts[1].split("Ellipse: ")[1]
                    elif key == 'Shape':
                        prefix = parts[0] + ',' + parts[1] + ',' + parts[2].split("Shape: ")[0] + "Shape: "
                        number = parts[2].split("Shape: ")[1]
                    (prefix_width, _), _ = cv2.getTextSize(prefix, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    (num_width, num_height), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    top_left = (x_text + prefix_width, cy - num_height)
                    bottom_right = (x_text + prefix_width + num_width, cy + 4)
                    cv2.rectangle(display, top_left, bottom_right, (255, 0, 0), 1)

        best_ids = set([hull_best[0], ellipse_best[0], shape_best[0]])
        for i, _, (x, y, w, h) in info_texts:
            if i in best_ids:
                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(display, f"Frame {idx+1}/{num}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        cv2.imshow(window, display)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
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
