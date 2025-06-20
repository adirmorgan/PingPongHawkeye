import json
import argparse
import cv2
import numpy as np
from typing import List, Tuple


def Color_Detection(frames: np.ndarray, frame_index: int, contour: np.ndarray, cfg: dict) -> float:
    """
    Compute a color score for the region inside the contour.
    Score is normalized mean V-channel brightness (0.0 to 1.0),
    only if the region is inside the ROI, V is high enough, and S is low enough.

    cfg keys:
      "min_v_brightness": int
      "max_s_saturation": int
      "roi_bounds": [x_min, x_max, y_min, y_max]
    """
    frame = frames[frame_index]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    # בדיקת מיקום הקונטור מול תחום ה-ROI
    x, y, w, h = cv2.boundingRect(contour)
    x_min, x_max, y_min, y_max = cfg.get("roi_bounds", [0, frame.shape[1], 0, frame.shape[0]])

    if not (x_min <= x <= x_max and y_min <= y <= y_max):
        return 0.0

    # מסכה פנימית של הקונטור
    mask = np.zeros_like(v_channel, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    vals_v = v_channel[mask == 255]
    vals_s = s_channel[mask == 255]

    if vals_v.size == 0 or vals_s.size == 0:
        return 0.0

    mean_v = float(np.mean(vals_v))
    mean_s = float(np.mean(vals_s))

    min_v = cfg.get('min_v_brightness', 200)
    max_s = cfg.get('max_s_saturation', 30)

    if mean_v < min_v or mean_s > max_s:
        return 0.0

    return mean_v / 255.0


def detect_ball_contours(frame: np.ndarray, cfg: dict):
    # 1) convert to HSV and threshold for white
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg['lower_hsv'], dtype=np.uint8)
    upper = np.array(cfg['upper_hsv'], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)

    # 2) clean up small holes and noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # 3) find and filter contours by size & ROI
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_min, x_max, y_min, y_max = cfg['roi_bounds']
    min_s, max_s = cfg['min_size'], cfg['max_size']

    valid = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue
        if not (min_s <= w <= max_s and min_s <= h <= max_s):
            continue
        valid.append(cnt)

    return valid, mask


def main():
    parser = argparse.ArgumentParser(
        description="Standalone test for color detection with contour scoring and pause control"
    )
    parser.add_argument('config', help='Path to JSON config file')
    parser.add_argument('--export-config', dest='export_config', help='Optional path to re-export full config')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        full_cfg = json.load(f)
    cfg = full_cfg['color_detection']

    frames = np.load(cfg['video_npy'])
    window = cfg.get('color_window', 'Color Detection')
    mask_window = 'Threshold Mask'
    delay = int(cfg.get('display_fps_delay', 30))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)
    idx = 0
    num = len(frames)
    paused = False

    while idx < num:
        frame = frames[idx]
        display = frame.copy()
        contours, mask = detect_ball_contours(frame, cfg)

        # debug: print count
        print(f"Frame {idx}: found {len(contours)} contours")

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            score = Color_Detection(frames, idx, cnt, cfg)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(display, f"{score:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow(window, display)
        cv2.imshow(mask_window, mask)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar toggles pause
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
