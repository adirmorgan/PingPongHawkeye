import json
import argparse
import cv2
import numpy as np


def compute_kstep_mask(frames: np.ndarray, frame_index: int, k: int, threshold: float) -> np.ndarray:
    """
    Compute binary motion mask via k-frame difference on grayscale.
    """
    if frame_index < k:
        return np.zeros(frames[0].shape[:2], dtype=np.uint8)
    curr = cv2.cvtColor(frames[frame_index], cv2.COLOR_BGR2GRAY).astype(np.float32)
    prev = cv2.cvtColor(frames[frame_index - k], cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = cv2.absdiff(curr, prev)
    mask = (diff > threshold).astype(np.uint8) * 255
    return mask


def compute_background_model(frames: np.ndarray, frame_index: int, learning_rate: float) -> np.ndarray:
    """
    Build a running background model up to frame_index using exponential averaging.
    """
    background = frames[0].astype('float32')
    for i in range(1, frame_index):
        cv2.accumulateWeighted(frames[i], background, learning_rate)
    return cv2.convertScaleAbs(background)


def Motion_Detection(frames: np.ndarray, frame_index: int, contour: np.ndarray, cfg: dict) -> float:
    """
    Compute a proximity-weighted motion score for a contour.

    cfg keys:
      "method": "kstep" or "background"
      "motion_k": int, "motion_threshold": float
      "background_learning_rate": float, "background_threshold": float
      "proximity_sigma": float
    """
    # Generate raw motion mask
    method = cfg.get('method', 'kstep')
    if method == 'kstep':
        k = cfg.get('motion_k', 5)
        thr = cfg.get('motion_threshold', 25.0)
        mask = compute_kstep_mask(frames, frame_index, k, thr)
    elif method == "background":
        lr = cfg.get('background_learning_rate', 0.01)
        thr = cfg.get('background_threshold', 30)
        bg = compute_background_model(frames, frame_index, lr)
        diff = cv2.absdiff(frames[frame_index], bg)
        b, g, r = cv2.split(diff)
        _, mb = cv2.threshold(b, thr, 255, cv2.THRESH_BINARY)
        _, mg = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)
        _, mr = cv2.threshold(r, thr, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(cv2.bitwise_or(mb, mg), mr)
    else :
        print("Error : method value is invalid")
        exit(1)
    # Create region mask for contour
    region = np.zeros_like(mask)
    cv2.drawContours(region, [contour], -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(region == 255)
    if ys.size == 0:
        return 0.0

    # Contour centroid for distance weighting
    moments = cv2.moments(contour)
    if moments.get('m00', 0) == 0:
        return 0.0
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']

    # Proximity weighting
    sigma = cfg.get('proximity_sigma', 10.0)
    d2 = (xs - cx)**2 + (ys - cy)**2
    weights = np.exp(-d2 / (2 * sigma**2))

    # Motion values normalized to [0,1]
    mvals = mask[ys, xs].astype(np.float32) / 255.0

    # Weighted average
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
        mask = (compute_kstep_mask(frames, idx, cfg.get('motion_k',5), cfg.get('motion_threshold',25.0))
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
