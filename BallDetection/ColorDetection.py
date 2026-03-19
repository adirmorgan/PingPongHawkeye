import argparse

import cv2
import numpy as np

import utils
from utils import *

@timeit("Color Detection")
def Color_Detection(frames: np.ndarray, frame_index: int, contour: np.ndarray, cfg: dict) -> float:
    """
    Compute a color score for the entire surface inside the contour.
    The calculation considers the full region enclosed by the contour.
    """
    frame = frames[frame_index]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the contour
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Compute mean HSV inside the contour
    mean_hsv = cv2.mean(hsv, mask=mask)[:3]  # (H, S, V)

    best = cfg["ideal_hsv"]
    lower = cfg["lower_hsv"]
    upper = cfg["upper_hsv"]

    if best == []:
        best = [np.mean([lower[i], upper[i]]) for i in range(3)]

    # Score each HSV channel
    scores = np.array([
        utils.piecewise_linear_scoring(lower[i], best[i], upper[i], mean_hsv[i])
        for i in range(len(['h', 's', 'v']))
    ], dtype=float)

    weights = cfg.get("weights")
    if weights is None:
        weights = [1, 1, 1]

    weights = np.array(weights, dtype=float)
    score = float(np.sum(scores * weights) / np.sum(weights))

    return score


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


    return contours, mask

def main():
    parser = argparse.ArgumentParser(
        description="Standalone test for color detection with contour scoring and pause control"
    )
    parser.add_argument('config', help='Path to JSON config file')
    parser.add_argument('--export-config', dest='export_config', help='Optional path to re-export full config')
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        full_cfg = json.load(f)

    cfg = full_cfg.get("color_detection", None)
    if cfg is None:
        raise ValueError("missing 'color_detection' configuration")

    timing(full_cfg['timing'])
    frames = np.load(cfg['video_npy'])
    window = cfg.get('color_window', 'Color Detection')
    mask_window = 'Threshold Mask'
    delay = int(cfg.get('display_fps_delay', 30))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)

    flow = video_flow_controller(nframes=len(frames), delay=delay)
    while flow.loop_cond():
        idx = flow.get_frame_index()
        frame = frames[idx]
        display = frame.copy()
        contours, mask = detect_ball_contours(frame, cfg)

        # debug: print count
        print(f"Frame {idx}: found {len(contours)} contours")

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            score = Color_Detection(frames, idx, cnt, cfg)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(display, f"{score:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # display the GUI's infoq
        info_text = flow.info_text()
        cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2 )
        # Show both the detection window and the mask window (from which the candidate contours were extracted)
        cv2.imshow(window, display)
        cv2.imshow(mask_window, mask)
        
        # Move to the next frame
        flow.next_frame()

    cv2.destroyAllWindows()
    if args.export_config:
        with open(args.export_config, 'w', encoding='utf-8') as ef:
            json.dump(full_cfg, ef, indent=4)
        print(f"Config exported to {args.export_config}")

if __name__ == '__main__':
    main()
