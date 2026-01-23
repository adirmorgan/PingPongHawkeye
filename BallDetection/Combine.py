#!/usr/bin/env python3
import json
import argparse
import numpy as np
import cv2
import concurrent.futures

from BallDetection import MotionDetection, ShapeDetection, ColorDetection
from utils import *  # assumes timeit, Contours, get_coordinates exist


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
           full_cfg: dict,
           executor: concurrent.futures.Executor | None = None
           ) -> tuple[np.ndarray | None,
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

    Parallelization (TODO applied):
      - For EACH contour, Shape/Color/Motion run concurrently in threads.

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
        contours = Contours(frames, frame_index, full_cfg)

    created_local_executor = False
    if executor is None:
        # Default: 3 workers is enough because we only parallelize 3 tasks per contour
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        created_local_executor = True

    try:
        for idx, contour in enumerate(contours):
            with timeit(f"Contour {idx}"):

                # Wrap each method so your timeit blocks still show meaningful labels
                def _shape():
                    with timeit("Shape Detection"):
                        return float(ShapeDetection.Shape_Detection(frames, frame_index, contour, shape_cfg))

                def _color():
                    with timeit("Color Detection"):
                        return float(ColorDetection.Color_Detection(frames, frame_index, contour, color_cfg))

                def _motion():
                    with timeit("Motion Detection"):
                        return float(MotionDetection.Motion_Detection(frames, frame_index, contour, motion_cfg))

                # --- Parallel run of the three detection methods
                fut_s = executor.submit(_shape)
                fut_c = executor.submit(_color)
                fut_m = executor.submit(_motion)

                # Collect (exceptions propagate here, which is good)
                s_score = fut_s.result()
                c_score = fut_c.result()
                m_score = fut_m.result()

                with timeit("Combining scores"):
                    combined_score = combine_scores(strategy, s_score, c_score, m_score, s_w, c_w, m_w)

                if combined_score > best_score:
                    best_s_score = s_score
                    best_c_score = c_score
                    best_m_score = m_score
                    best_score = combined_score
                    best_contour = contour

    finally:
        if created_local_executor:
            executor.shutdown(wait=True)

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

    if out.dtype != np.uint8:
        out = np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0)
        out = np.clip(out, 0, 255)
        if out.max() <= 1.0:
            out = out * 255.0
        out = out.astype(np.uint8)

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

    with open(args.config, "r", encoding="utf-8") as f:
        full_cfg = json.load(f)

    combine_cfg = full_cfg["combine"]

    timing(full_cfg.get("timing", False))

    npy_path = combine_cfg["npy_file"]
    frames = np.load(npy_path)

    window_name = combine_cfg.get("window_name", "combined_score Detection")
    delay = int(combine_cfg.get("display_fps_delay", 30))
    show_components = bool(combine_cfg.get("show_components", False))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    coords: list[tuple[int, int] | None] = []

    # Create ONE executor and reuse it for all frames (lower overhead).
    # Only 3 workers are needed because we parallelize exactly 3 detection tasks per contour.
    workers = int(combine_cfg.get("detectors_workers", 3))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for frame_idx in range(frames.shape[0]):
            frame = frames[frame_idx]
            display = to_displayable(frame).copy()

            best_contour, best_coord, combined_score, s, c, m = TOP_2D(frames, frame_idx, full_cfg, executor=executor)
            coords.append(best_coord)

            if best_contour is not None and len(best_contour) > 0:
                if best_contour.dtype not in (np.float32, np.int32):
                    best_contour = best_contour.astype(np.float32)

                x, y, w, h = cv2.boundingRect(best_contour)
                cv2.drawContours(display, [best_contour], -1, (0, 255, 0), 1)

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

            cv2.imshow(window_name, display)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()

    if args.export_coords:
        with open(args.export_coords, "w", encoding="utf-8") as of:
            json.dump({"coordinates": coords}, of, indent=4)
        print(f"2D coordinates exported to {args.export_coords}")


if __name__ == "__main__":
    main()
