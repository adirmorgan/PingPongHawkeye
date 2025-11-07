import argparse

import utils
from utils import *

def Color_Detection(frames: np.ndarray, frame_index: int, contour: np.ndarray, cfg: dict) -> float:
    """
    Compute a color score for the entire surface inside the contour.
    Score is normalized mean V-channel brightness (0.0 to 1.0).
    The calculation considers the full region enclosed by the contour.
    cfg keys:
      "min_v_brightness": int
    """
    frame = frames[frame_index]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mask = np.zeros_like(v, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)  # Fill the entire contour area for accurate scoring
    vals = v[mask == 255]  # Brightness values from the full surface enclosed by the contour
    if vals.size == 0:
        return 0.0
    mean_v = float(np.mean(vals))
    if mean_v < cfg.get('min_v_brightness', 0):
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
