#!/usr/bin/env python3
import json
import argparse
import numpy as np
import cv2

from BallDetection import MotionDetection, ShapeDetection, ColorDetection
from utils import *  # assumes timeit, timing, Contours, get_coordinates exist


def combine_scores(strategy: str,
                   s_score: float,
                   c_score: float,
                   m_score: float,
                   s_w: float,
                   c_w: float,
                   m_w: float) -> float:
    """
    Combine three scores according to the selected strategy.
    All scores are assumed to be in [0, 1].
    """
    match strategy:
        case "augment":   # weighted RMS
            return (s_w * s_score ** 2 + c_w * c_score ** 2 + m_w * m_score ** 2) ** 0.5
        case "maximize":
            return max(s_score, c_score, m_score)
        case "gather":    # weighted arithmetic mean
            return s_score * s_w + c_score * c_w + m_score * m_w
        case "filter":    # weighted geometric mean
            total_w = s_w + c_w + m_w
            if total_w <= 0:
                return 0.0
            # Avoid log/zero issues by clipping
            s = max(s_score, 1e-8)
            c = max(c_score, 1e-8)
            m = max(m_score, 1e-8)
            return (s ** s_w * c ** c_w * m ** m_w) ** (1.0 / total_w)
        case "minimize":
            return min(s_score, c_score, m_score)
        case "diminish":  # weighted harmonic mean
            denom = 0.0
            if s_score > 0:
                denom += s_w / s_score
            if c_score > 0:
                denom += c_w / c_score
            if m_score > 0:
                denom += m_w / m_score
            return 1.0 / denom if denom > 0 else 0.0
        case _:
            return 0.0


@timeit("TOP_2D")
def TOP_2D(frames: np.ndarray,
           frame_index: int,
           full_cfg: dict) -> tuple[np.ndarray | None,
                                    tuple[int, int] | None,
                                    float,
                                    float,
                                    float,
                                    float]:
    """
    Process a single frame:
      - Extract candidate contours.
      - Compute shape/color/motion scores for each.
      - Combine scores according to config.
    Returns:
      best_contour: best contour or None
      best_coord: (x, y) of best contour's centroid or None
      best_score: combined score or 0.0
      best_s_score, best_c_score, best_m_score: component scores for best contour
    """
    with timeit("Setup"):
        combine_cfg = full_cfg["combine"]
        shape_cfg = full_cfg["shape_detection"]
        color_cfg = full_cfg["color_detection"]
        motion_cfg = full_cfg["motion_detection"]

        s_w = float(combine_cfg["shape_weight"])
        c_w = float(combine_cfg["color_weight"])
        m_w = float(combine_cfg["motion_weight"])
        min_score = float(combine_cfg.get("min_score", 0.0))
        strategy = combine_cfg.get("strategy", "gather")

    best_score = min_score
    best_s_score, best_c_score, best_m_score = 0.0, 0.0, 0.0
    best_contour = None

    with timeit("Contours"):
        # Contours() is expected to use frames[frame_index] internally with shape config
        contours = Contours(frames, frame_index, full_cfg)

    for idx, contour in enumerate(contours):
        with timeit(f"Contour {idx}"):
            with timeit("Shape Detection"):
                s_score = float(ShapeDetection.Shape_Detection(frames, frame_index, contour, shape_cfg))
            with timeit("Color Detection"):
                c_score = float(ColorDetection.Color_Detection(frames, frame_index, contour, color_cfg))
            with timeit("Motion Detection"):
                m_score = float(MotionDetection.Motion_Detection(frames, frame_index, contour, motion_cfg))

            with timeit("Combining scores"):
                combined_score = combine_scores(strategy, s_score, c_score, m_score, s_w, c_w, m_w)

            if combined_score > best_score:
                best_s_score = s_score
                best_c_score = c_score
                best_m_score = m_score
                best_score = combined_score
                best_contour = contour

    best_coord = get_coordinates(best_contour) if best_contour is not None else None
    if best_coord is None:
        best_score = 0.0

    return best_contour, best_coord, best_score, best_s_score, best_c_score, best_m_score


def to_displayable(img: np.ndarray) -> np.ndarray:
    """
    Ensure img is uint8 and 3-channel BGR for visualization + drawing colored overlays.
    Handles common cases: float images in [0,1] or [0,255], grayscale frames, etc.
    """
    out = img

    # Convert dtype to uint8
    if out.dtype != np.uint8:
        out = np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0)
        out = np.clip(out, 0, 255)

        # If looks normalized, scale up
        if out.max() <= 1.0:
            out = out * 255.0

        out = out.astype(np.uint8)

    # Ensure 3 channels (BGR)
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif out.ndim == 3 and out.shape[2] == 1:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Combine Motion/Shape/Color to track ping-pong ball (2D) with GUI"
    )
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument(
        "--export-coords",
        dest="export_coords",
        help="Optional path to save 2D coordinates JSON"
    )
    args = parser.parse_args()

    # Load full configuration
    with open(args.config, "r", encoding="utf-8") as f:
        full_cfg = json.load(f)

    combine_cfg = full_cfg["combine"]

    # Global timing toggle
    timing(full_cfg.get("timing", False))

    # Load frames
    npy_path = combine_cfg["npy_file"]
    frames = np.load(npy_path)

    # Visualization params
    window_name = combine_cfg.get("window_name", "combined_score Detection")
    delay = int(combine_cfg.get("display_fps_delay", 30))
    show_components = bool(combine_cfg.get("show_components", False))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    coords: list[tuple[int, int] | None] = []

    for frame_idx in range(frames.shape[0]):
        frame = frames[frame_idx]
        display = to_displayable(frame).copy()

        best_contour, best_coord, combined_score, s, c, m = TOP_2D(frames, frame_idx, full_cfg)
        coords.append(best_coord)

        if best_contour is not None and len(best_contour) > 0:
            # Validate contour data type
            if best_contour.dtype not in (np.float32, np.int32):
                best_contour = best_contour.astype(np.float32)

            # Compute bounding rectangle
            x, y, w, h = cv2.boundingRect(best_contour)
            cv2.drawContours(display, [best_contour], -1, (0, 255, 0), 1)

            # Display scores
            text = f"{combined_score:.2f} S:{s:.2f} C:{c:.2f} M:{m:.2f}" if show_components else f"{combined_score:.2f}"
            cv2.putText(
                display,
                text,
                (x, max(y - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        else:
            # Not an error; just means nothing passed the threshold
            pass

        # ✅ Actually show the frame (this was missing in your code)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    # Optional export of coordinates
    if args.export_coords:
        with open(args.export_coords, "w", encoding="utf-8") as of:
            json.dump({"coordinates": coords}, of, indent=4)
        print(f"2D coordinates exported to {args.export_coords}")


if __name__ == "__main__":
    main()
# TODO: try to make code parallel! using threads on the different detection methods