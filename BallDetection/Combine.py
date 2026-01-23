#!/usr/bin/env python3
import argparse
import json
import cv2
import numpy as np
import concurrent.futures

from BallDetection import MotionDetection, ShapeDetection, ColorDetection
from utils import *  # assumes timeit, timing, Contours, get_coordinates, threadit, THREADING_ENABLE exist


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
def TOP_2D(
    frames: np.ndarray,
    frame_index: int,
    full_cfg: dict,
    contour_executor: concurrent.futures.Executor | None = None,
    method_executor: concurrent.futures.Executor | None = None,
) -> tuple[np.ndarray | None,
           tuple[int, int] | None,
           float,
           float,
           float,
           float]:
    """
    Parallelization (when enabled and pools provided):
      - Between contours: uses contour_executor
      - Between detection methods (S/C/M) per contour: uses method_executor

    If pools are None:
      - uses internal threadit pools (which become sequential if utils.threading(False) was called).
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

    with timeit("Contours"):
        contours = Contours(frames, frame_index, full_cfg)

    if not contours:
        return None, None, 0.0, 0.0, 0.0, 0.0

    use_external_contour = contour_executor is not None
    use_external_method = method_executor is not None
    same_pool = (use_external_contour and use_external_method and contour_executor is method_executor)

    def eval_methods_sequential(contour) -> tuple[float, float, float]:
        s_score = float(ShapeDetection.Shape_Detection(frames, frame_index, contour, shape_cfg))
        c_score = float(ColorDetection.Color_Detection(frames, frame_index, contour, color_cfg))
        m_score = float(MotionDetection.Motion_Detection(frames, frame_index, contour, motion_cfg))
        return s_score, c_score, m_score

    def eval_contour_with_method_pool(contour, method_pool) -> tuple[float, float, float, float, np.ndarray]:
        if same_pool:
            s_score, c_score, m_score = eval_methods_sequential(contour)
        else:
            fs = method_pool.submit(ShapeDetection.Shape_Detection, frames, frame_index, contour, shape_cfg)
            fc = method_pool.submit(ColorDetection.Color_Detection, frames, frame_index, contour, color_cfg)
            fm = method_pool.submit(MotionDetection.Motion_Detection, frames, frame_index, contour, motion_cfg)

            s_score = float(fs.result())
            c_score = float(fc.result())
            m_score = float(fm.result())

        combined = combine_scores(strategy, s_score, c_score, m_score, s_w, c_w, m_w)
        return float(combined), float(s_score), float(c_score), float(m_score), contour

    best_score = min_score
    best_s = best_c = best_m = 0.0
    best_contour = None

    # ---- External pools path (used by TOP_3D and by this file's main() when THREADING_ENABLE)
    if use_external_contour and use_external_method:
        futures = [contour_executor.submit(eval_contour_with_method_pool, contour, method_executor)
                   for contour in contours]

        for fut in concurrent.futures.as_completed(futures):
            combined, s, c, m, contour = fut.result()
            if combined > best_score:
                best_score = combined
                best_s, best_c, best_m = s, c, m
                best_contour = contour

    # ---- Internal threadit path (toggleable; threadit is sequential when THREADING_ENABLE=False)
    else:
        contour_workers = int(full_cfg.get("combine", {}).get("contour_workers", min(32, len(contours))))
        method_workers = int(full_cfg.get("combine", {}).get("method_workers", 3 * max(1, min(8, contour_workers))))

        with threadit(max_workers=contour_workers) as contour_th, threadit(max_workers=method_workers) as method_th:
            futs = [contour_th.submit(eval_contour_with_method_pool, contour, method_th)
                    for contour in contours]

            for fut in contour_th.as_completed(futs):
                combined, s, c, m, contour = fut.result()
                if combined > best_score:
                    best_score = combined
                    best_s, best_c, best_m = s, c, m
                    best_contour = contour

    best_coord = get_coordinates(best_contour) if best_contour is not None else None
    if best_coord is None:
        best_score = 0.0

    return best_contour, best_coord, best_score, best_s, best_c, best_m


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

    # Reuse pools across frames when threading is enabled
    contour_workers = int(combine_cfg.get("contour_workers", 16))
    method_workers = int(combine_cfg.get("method_workers", 3 * max(1, contour_workers)))

    contour_executor = None
    method_executor = None
    created = False

    if THREADING_ENABLE:
        contour_executor = concurrent.futures.ThreadPoolExecutor(max_workers=contour_workers)
        method_executor = concurrent.futures.ThreadPoolExecutor(max_workers=method_workers)
        created = True

    try:
        for frame_idx in range(frames.shape[0]):
            frame = frames[frame_idx]
            display = to_displayable(frame).copy()

            best_contour, best_coord, combined_score, s, c, m = TOP_2D(
                frames,
                frame_idx,
                full_cfg,
                contour_executor=contour_executor,
                method_executor=method_executor,
            )
            coords.append(best_coord)

            if best_contour is not None and len(best_contour) > 0:
                if best_contour.dtype not in (np.float32, np.int32):
                    best_contour = best_contour.astype(np.float32)

                x, y, w, h = cv2.boundingRect(best_contour)
                cv2.drawContours(display, [best_contour], -1, (0, 255, 0), 1)

                text = (f"{combined_score:.2f} S:{s:.2f} C:{c:.2f} M:{m:.2f}"
                        if show_components else f"{combined_score:.2f}")
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

    finally:
        if created:
            contour_executor.shutdown(wait=True)
            method_executor.shutdown(wait=True)

    cv2.destroyAllWindows()

    if args.export_coords:
        with open(args.export_coords, "w", encoding="utf-8") as of:
            json.dump({"coordinates": coords}, of, indent=4)
        print(f"2D coordinates exported to {args.export_coords}")


if __name__ == "__main__":
    main()
