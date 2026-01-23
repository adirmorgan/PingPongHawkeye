import json
import argparse
import cv2
import numpy as np
from utils import *

# TODO : kstep takes too much time... major bottleneck of our entire program!
#   probably due to making the musk of the entire frame, again and again per each contour...
#   sol1 - make musk per contour.
#   sol2 - make musk for the entire frame only for the first contour, and reuse it for later contours

def compute_kstep_motion(frames: np.ndarray,
                         frame_index: int,
                         k: int,
                         threshold: float,
                         shadow_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute:
      - binary motion mask via k-frame difference
      - continuous motion_strength in [0,1], where shadows are down-weighted.

    shadow_weight < 1.0 reduces the contribution of darkening (shadows).
    """
    h, w = frames[0].shape[:2]

    if frame_index < k:
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.float32)

    # Current and temporal average of previous k frames in grayscale
    curr = cv2.cvtColor(frames[frame_index], cv2.COLOR_BGR2GRAY).astype(np.float32)
    prev_stack = [
        cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        for i in range(frame_index - k, frame_index)
    ]
    base = sum(prev_stack) / float(k)

    # Directional difference
    diff = curr - base
    pos = np.clip(diff, 0, None)        # brightening (object)
    neg = np.clip(-diff, 0, None)       # darkening (shadow)

    # Shadows contribute less
    motion = (1-shadow_weight)*pos + shadow_weight * neg  # still >= 0

    # Binary mask for visualization / contour extraction
    mask = (motion > threshold).astype(np.uint8) * 255

    # Smooth and dilate a bit
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Normalize motion to [0,1] for scoring
    max_val = motion.max()
    if max_val > 0:
        motion_strength = (motion / max_val).astype(np.float32)
    else:
        motion_strength = np.zeros_like(motion, dtype=np.float32)

    return mask, motion_strength


def compute_background_model(frames: np.ndarray,
                             frame_index: int,
                             learning_rate: float) -> np.ndarray:
    """
    Build a running background model up to frame_index using exponential averaging.
    """
    background = frames[0].astype(np.float32)
    for i in range(1, frame_index):
        cv2.accumulateWeighted(frames[i], background, learning_rate)
    return cv2.convertScaleAbs(background)


def Motion_Detection(frames: np.ndarray,
                     frame_index: int,
                     contour: np.ndarray,
                         cfg: dict) -> float:
        """
        Compute a proximity-weighted motion score for a contour.

        cfg keys:
          - "method": "kstep" or "background"
          - "motion_k": int
          - "motion_threshold": float
          - "background_learning_rate": float
          - "background_threshold": float
          - "proximity_sigma": float
          - "shadow_weight": float in [0,1], how much to trust darkening (shadows)
        """
        with timeit("first"):
            method = cfg.get("method", "kstep")
            shadow_weight = float(cfg.get("shadow_weight", 0.3))

        with timeit("middle"):
            if method == "kstep":
                k = int(cfg.get("motion_k", 5))
                thr = float(cfg.get("motion_threshold", 25.0))
                mask, motion_strength = compute_kstep_motion(frames, frame_index, k, thr, shadow_weight)

            elif method == "background":
                lr = float(cfg.get("background_learning_rate", 0.01))
                thr = float(cfg.get("background_threshold", 30.0))
                bg = compute_background_model(frames, frame_index, lr)

                diff = cv2.absdiff(frames[frame_index], bg)
                b, g, r = cv2.split(diff)
                _, mb = cv2.threshold(b, thr, 255, cv2.THRESH_BINARY)
                _, mg = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)
                _, mr = cv2.threshold(r, thr, 255, cv2.THRESH_BINARY)
                mask = cv2.bitwise_or(cv2.bitwise_or(mb, mg), mr)

                # For background mode: simple normalized magnitude as motion_strength
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32)
                max_val = gray_diff.max()
                if max_val > 0:
                    motion_strength = (gray_diff / max_val).astype(np.float32)
                else:
                    motion_strength = np.zeros_like(gray_diff, dtype=np.float32)

            else:
                print("Error: method value is invalid")
                return 0.0

        with timeit("last"):
            # Region mask for this contour
            region = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(region, [contour], -1, 255, thickness=cv2.FILLED)

            ys, xs = np.where(region == 255)
            if ys.size == 0:
                return 0.0

            # Contour centroid
            moments = cv2.moments(contour)
            if moments.get("m00", 0) == 0:
                return 0.0
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]

            # Proximity weighting
            sigma = float(cfg.get("proximity_sigma", 10.0))
            d2 = (xs - cx) ** 2 + (ys - cy) ** 2
            weights = np.exp(-d2 / (2.0 * sigma ** 2))
            weights /= weights.sum() #  Normalize weights

            # Sample continuous motion strength inside contour
            mvals = motion_strength[ys, xs]  # already in [0,1]
            mvals = np.clip(2*mvals, 0, 1)
            #approximately half of the color diff would be outside of the contour... only inward diff is accounted- but there is the back

            # Weighted average motion score
            weighted_sum = np.sum(weights * mvals)
            weight_total = np.sum(weights)

            return float(weighted_sum / weight_total) if weight_total > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Motion detection with proximity-weighted scoring")
    parser.add_argument('config', help='Path to JSON config file')
    parser.add_argument('--export-config', dest='export_config', help='Optional path to re-export config')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)['motion_detection']

    frames = np.load(cfg['video_npy'])
    window = cfg.get('motion_window', 'Motion Detection')
    delay = int(cfg.get('display_fps_delay', 30))
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    for idx in range(len(frames)):
        frame = frames[idx]
        mask = (compute_kstep_motion(frames, idx, cfg.get('motion_k',5), cfg.get('motion_threshold',25.0), shadow_weight=cfg.get('shadow_weight',0.3))[0]
                if cfg.get('method','kstep')=='kstep'
                else None)  # background mask can be computed similarly
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        display = frame.copy()
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            score = Motion_Detection(frames, idx, cnt, cfg)
            cv2.rectangle(display, (x,y), (x+w,y+h), (255,0,0),2)
            cv2.putText(display, f"{score:.2f}", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        cv2.imshow(window, display)
        cv2.imshow('Mask', mask)
        if cv2.waitKey(delay)&0xFF==ord('q'):
            break

    cv2.destroyAllWindows()
    if args.export_config:
        with open(args.export_config,'w',encoding='utf-8') as ef:
            json.dump(cfg, ef, indent=4)
        print(f"Config exported to {args.export_config}")

if __name__=='__main__':
    main()
