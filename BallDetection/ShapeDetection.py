import json
import argparse
import cv2
import numpy as np
from typing import List, Tuple

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
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    return mask

def Shape_Detection(frames: np.ndarray, frame_index: int, contour: np.ndarray, cfg: dict) -> float:
    """
    Compute a composite elliptical score for a contour.
    Combines axis ratio and mean normalized distance to ellipse boundary.
    Filters by contour area constraints.
    """
    frame = frames[frame_index]

    # filter by area
    area = cv2.contourArea(contour)
    min_a = cfg.get('min_area', 0)
    max_a = cfg.get('max_area', float('inf'))
    if area < min_a or area > max_a:
        return 0.0

    if len(contour) < 5:
        return 0.0

    # fit ellipse
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (major, minor), angle = ellipse
    if major <= 0 or minor <= 0:
        return 0.0

    # axis ratio
    axis_ratio = min(major, minor) / max(major, minor)

    # mask for ellipse boundary distance
    h, w = frame.shape[:2]
    ellipse_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ellipse, 255, -1)

    # distance transforms: inside and outside
    inv_mask = cv2.bitwise_not(ellipse_mask)
    dt_outside = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
    dt_inside  = cv2.distanceTransform(ellipse_mask, cv2.DIST_L2, 5)

    # collect distances for contour points
    distances = []
    for pt in contour.squeeze():
        x_pt, y_pt = int(pt[0]), int(pt[1])
        if 0 <= x_pt < w and 0 <= y_pt < h:
            if ellipse_mask[y_pt, x_pt] > 0:
                d = dt_inside[y_pt, x_pt]
            else:
                d = dt_outside[y_pt, x_pt]
            distances.append(d)
    if not distances:
        # no valid points sampled
        return axis_ratio

    # normalize by ellipse boundary radius (use larger radius)
    radius = max(major, minor) / 2.0
    mean_norm_dist = float(np.mean(distances)) / radius

    # composite score
    w = cfg.get('ellipse_weight', 0.5)
    composite = (1 - w) * axis_ratio + w * (1 - mean_norm_dist)
    return float(np.clip(composite, 0.0, 1.0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standalone test for shape detection scoring")
    parser.add_argument('config', help='Path to JSON config file')
    parser.add_argument('--export-config', dest='export_config', help='Optional path to save config')
    args = parser.parse_args()

    cfg = json.load(open(args.config, 'r', encoding='utf-8'))['shape_detection']
    frames = np.load(cfg['video_npy'])
    window = cfg.get('shape_window', 'Shape Detection')
    delay = int(cfg.get('display_fps_delay', 30))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    idx = 0
    while True:
        frame = frames[idx]
        mask = preprocess_mask(frame, cfg)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        display = frame.copy()
        for cnt in contours:
            score = Shape_Detection(frames, idx, cnt, cfg)
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            cv2.rectangle(display, (x, y), (x+w_rect, y+h_rect), (0, 255, 0), 2)
            cv2.putText(display, f"{score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow(window, display)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('d'), 83]:
            idx = (idx + 1) % len(frames)
        elif key in [ord('a'), 81]:
            idx = (idx - 1) % len(frames)

    cv2.destroyAllWindows()
    if args.export_config:
        with open(args.export_config, 'w', encoding='utf-8') as ef:
            json.dump({'shape_detection': cfg}, ef, indent=4)
        print(f"Config exported to {args.export_config}")
