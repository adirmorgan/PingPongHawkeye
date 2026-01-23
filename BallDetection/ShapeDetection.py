import argparse
from typing import Dict
from utils import *  # assumes timing()/timeit/time config exist


def preprocess_mask(frame: np.ndarray, cfg: Dict) -> np.ndarray:
    """
    Generate binary mask for candidate ball regions based on color/Lab thresholds.
    Morphology is tuned to avoid merging nearby objects too aggressively.
    cfg keys (examples):
      - "color_space": "HSV" or "YCrCb"
      - "lower_thresh": [H, S, V] or [Y, Cr, Cb]
      - "upper_thresh": [H, S, V] or [Y, Cr, Cb]
      - "lab_thresh": float (optional)
      - "lab_ref": [L, a, b] (optional)
      - "gaussian_ksize": int (odd)
      - "open_ksize": int
      - "close_ksize": int
    """
    # Ensure 3 channels (convert grayscale to BGR if needed)
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Optional Lab-based distance thresholding
    lab_thresh = cfg.get("lab_thresh", None)
    if lab_thresh is not None:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab).astype(np.float32)
        ref = np.array(cfg.get("lab_ref", [200.0, 128.0, 128.0]), dtype=np.float32)
        dist = np.linalg.norm(lab - ref[None, None, :], axis=2)
        mask = (dist < float(lab_thresh)).astype(np.uint8) * 255
    else:
        # Default: HSV or YCrCb in-range threshold
        space = cfg.get("color_space", "HSV").upper()
        if space == "HSV":
            conv = cv2.COLOR_BGR2HSV
        elif space in ("YCRCB", "YCrCb"):
            conv = cv2.COLOR_BGR2YCrCb
        else:
            conv = cv2.COLOR_BGR2HSV  # fallback

        cs = cv2.cvtColor(frame, conv)
        lower = np.array(cfg.get("lower_thresh", [0, 0, 200]), dtype=np.uint8)
        upper = np.array(cfg.get("upper_thresh", [180, 30, 255]), dtype=np.uint8)
        mask = cv2.inRange(cs, lower, upper)

    # Light blur to smooth noise, but not too large to avoid merging objects
    gaussian_ksize = int(cfg.get("gaussian_ksize", 5))
    if gaussian_ksize % 2 == 0:
        gaussian_ksize += 1
    if gaussian_ksize >= 3:
        mask = cv2.GaussianBlur(mask, (gaussian_ksize, gaussian_ksize), 0)

    # Morphology: first OPEN (remove noise), then a small CLOSE (fix small holes)
    open_ksize = int(cfg.get("open_ksize", 3))
    close_ksize = int(cfg.get("close_ksize", 3))

    if open_ksize > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    if close_ksize > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask

@timeit("Shape Detection")
def Shape_Detection(frames: np.ndarray, frame_index: int, contour: np.ndarray, cfg: Dict) -> float:
    """
    Compute a composite ellipse-based score for a contour.

    Logic:
      - Reject by area.
      - Fit ellipse.
      - Use axis ratio (close to 1 is better).
      - Use distance of contour points from fitted ellipse boundary.
      - Combine into a score in [0, 1].

    cfg keys:
      - "min_area": float
      - "max_area": float
      - "ellipse_weight": float in [0,1]
    """
    frame = frames[frame_index]

    # Area filter
    area = cv2.contourArea(contour)
    min_a = float(cfg.get("min_area", 0.0))
    max_a = float(cfg.get("max_area", float("inf")))
    if area < min_a or area > max_a:
        return 0.0

    if len(contour) < 5:
        return 0.0

    # Fit ellipse
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (major, minor), angle = ellipse
    if major <= 0 or minor <= 0:
        return 0.0

    # Axis ratio: 1.0 is perfect circle
    axis_ratio = min(major, minor) / max(major, minor)

    # Build filled ellipse mask
    h, w = frame.shape[:2]
    ellipse_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ellipse, 255, -1)

    # Distance transforms for inside/outside distances
    inv_mask = cv2.bitwise_not(ellipse_mask)
    dt_outside = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
    dt_inside = cv2.distanceTransform(ellipse_mask, cv2.DIST_L2, 5)

    # Collect distances for contour points
    distances = []
    contour_points = contour.squeeze()

    if contour_points.ndim == 1:
        contour_points = np.expand_dims(contour_points, axis=0)

    for pt in contour_points:
        x_pt, y_pt = int(pt[0]), int(pt[1])
        if 0 <= x_pt < w and 0 <= y_pt < h:
            if ellipse_mask[y_pt, x_pt] > 0:
                d = dt_inside[y_pt, x_pt]
            else:
                d = dt_outside[y_pt, x_pt]
            distances.append(d)

    if not distances:
        return axis_ratio

    # Normalize mean distance by ellipse "radius"
    radius = max(major, minor) / 2.0
    if radius <= 0:
        return 0.0

    mean_norm_dist = float(np.mean(distances)) / radius

    # Composite score
    ellipse_weight = float(cfg.get("ellipse_weight", 0.5))
    ellipse_weight = max(0.0, min(1.0, ellipse_weight))

    score = (1.0 - ellipse_weight) * axis_ratio + ellipse_weight * (1.0 - mean_norm_dist)
    score = float(np.clip(score, 0.0, 1.0))

    return score




def main():
    parser = argparse.ArgumentParser(description="Shape-based ping-pong ball detection")
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument(
        "--export-config",
        dest="export_config",
        help="Optional path to export the used Shape Detection config"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        full_cfg = json.load(f)

    cfg = full_cfg.get("shape_detection", None)
    if cfg is None:
        raise ValueError("missing 'shape_detection' configuration")

    timing(full_cfg.get("timing", False))

    # Load frames
    frames = np.load(cfg["video_npy"])

    window_name = cfg.get("window_name", "Shape Detection")
    mask_window_name = cfg.get("mask_window_name", "Shape Mask")
    delay = int(cfg.get("display_fps_delay", 30))
    min_score = float(cfg.get("thresh", 0.5))
    draw_ellipse = bool(cfg.get("draw_ellipse", True))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(mask_window_name, cv2.WINDOW_NORMAL)

    flow = video_flow_controller(nframes = len(frames), delay = delay)
    while(flow.loop_cond()):
        idx = flow.get_frame_index()
        frame = frames[idx]
        display = frame.copy()

        # 1) Build mask for candidate regions
        mask = preprocess_mask(frame, cfg)

        # 2) Extract contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 3) Evaluate and draw all contours above min_score
        for cnt in contours:
            score = Shape_Detection(frames, idx, cnt, cfg)
            if score >= min_score:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if draw_ellipse and len(cnt) >= 5:
                    ellipse = cv2.fitEllipse(cnt)
                    cv2.ellipse(display, ellipse, (255, 0, 0), 2)

                cv2.putText(
                    display,
                    f"{score:.2f}",
                    (x, max(y - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        # 4) display GUI info
            info_text = flow.info_text()
            cv2.putText(display, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # 5) Show windows
        cv2.imshow(window_name, display)
        cv2.imshow(mask_window_name, mask)

        # 6) move to the next frame
        flow.next_frame()

    cv2.destroyAllWindows()

    if args.export_config:
        with open(args.export_config, "w", encoding="utf-8") as ef:
            json.dump(cfg, ef, indent=4)
        print(f"Config exported to {args.export_config}")


if __name__ == "__main__":
    main()
