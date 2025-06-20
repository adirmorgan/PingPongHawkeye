import json
import argparse
import numpy as np
import cv2
from typing import Tuple, List

from BallDetection import MotionDetection, ShapeDetection, ColorDetection
from utils import get_coordinates, Contours

def TOP_2(npy_file_path: str, full_cfg: dict) -> Tuple[list, list]:
    """
    Process frames and return:
    - coords: list of best centroid coordinates per frame
    - all_scored: list of list of tuples (contour, score) per frame (only those above min_score)
    """
    combine_cfg = full_cfg['combine']
    shape_cfg = full_cfg['shape_detection']
    color_cfg = full_cfg['color_detection']
    motion_cfg = full_cfg['motion_detection']

    s_weight = combine_cfg['shape_weight']
    c_weight = combine_cfg['color_weight']
    m_weight = combine_cfg['motion_weight']

    min_score = combine_cfg.get('min_score', 0.0)
    strategy = combine_cfg.get('strategy', 'gather')

    frames = np.load(npy_file_path)
    coords = []
    all_scored = []  # ← כאן נאחסן את כל הקונטורים עם ציונים מעל סף

    for idx in range(len(frames)):
        best_contour = None
        best_score = min_score
        frame_scores = []

        candidates = Contours(frames, idx, shape_cfg)
        for contour, _ in candidates:
            pt = get_coordinates(contour)
            if pt is None:
                continue

            s_score = ShapeDetection.shape_score(contour, shape_cfg)
            c_score = ColorDetection.Color_Detection(frames, idx, contour, color_cfg)
            m_score = MotionDetection.Motion_Detection(frames, idx, contour, motion_cfg)

            # combine strategy
            if strategy == 'augment':
                score = np.sqrt(s_weight*s_score**2 + c_weight*c_score**2 + m_weight*m_score**2)
            elif strategy == 'maximize':
                score = max(s_score, c_score, m_score)
            elif strategy == 'gather':
                score = s_weight*s_score + c_weight*c_score + m_weight*m_score
            elif strategy == 'filter':
                score = s_score**s_weight * c_score**c_weight * m_score**m_weight
            elif strategy == 'minimize':
                score = min(s_score, c_score, m_score)
            elif strategy == 'diminish':
                score = 1 / (s_weight/s_score + c_weight/c_score + m_weight/m_score)
            else:
                print("Error: invalid combine-strategy")
                exit(1)

            if score > min_score:
                frame_scores.append((contour, score))
            if score > best_score:
                best_score = score
                best_contour = contour

        all_scored.append(frame_scores)
        coords.append(get_coordinates(best_contour) if best_contour is not None else None)

        if combine_cfg.get('print', False):
            print(f"Frame {idx}: coordinates = {coords[-1]}")

    return coords, all_scored

def TOP(npy_file_path: str, full_cfg: dict) -> list:
    """
    Process frames and return list of centroid coordinates per frame.
    full_cfg keys:
      "combine", "shape_detection", "color_detection", "motion_detection"
    """
    combine_cfg = full_cfg['combine']
    shape_cfg = full_cfg['shape_detection']
    color_cfg = full_cfg['color_detection']
    motion_cfg = full_cfg['motion_detection']

    # weights
    s_weight = combine_cfg['shape_weight']
    c_weight = combine_cfg['color_weight']
    m_weight = combine_cfg['motion_weight']

    # configuration
    min_score = combine_cfg.get('min_score', 0.0)
    strategy = combine_cfg.get('strategy', 'gather')

    frames = np.load(npy_file_path)
    coords = []

    for idx in range(len(frames)):
        best_contour = None
        best_score = min_score

        # get all candidate contours via shape
        candidates = Contours(frames, idx, shape_cfg)
        for contour, _ in candidates:
            pt = get_coordinates(contour)
            if pt is None:
                continue

            # individual scores
            s_score = ShapeDetection.shape_score(contour, shape_cfg)
            c_score = ColorDetection.Color_Detection(frames, idx, contour, color_cfg)
            m_score = MotionDetection.Motion_Detection(frames, idx, contour, motion_cfg)

            # combine per strategy
            if strategy == 'augment':  # weighted quadratic mean
                score = np.sqrt(s_weight*s_score**2 + c_weight*c_score**2 + m_weight*m_score**2)
            elif strategy == 'maximize':
                score = max(s_score, c_score, m_score)
            elif strategy == 'gather':  # weighted average
                score = s_weight*s_score + c_weight*c_score + m_weight*m_score
            elif strategy == 'filter':  # weighted geometric mean
                score = (s_score**s_weight * c_score**c_weight * m_score**m_weight)
            elif strategy == 'minimize':
                score = min(s_score, c_score, m_score)
            elif strategy == 'diminish':  # weighted harmonic mean
                score = 1/(s_weight/s_score + c_weight/c_score + m_weight/m_score)
            else: # should never happen
                print("Error : invalid combine-strategy")
                exit(1);

            if score > best_score:
                best_score = score
                best_contour = contour

        coords.append(get_coordinates(best_contour) if best_contour is not None else None)
        if combine_cfg.get('print', False):
            print(f"Frame {idx}: coordinates = {coords[-1]}")

    return coords


def display_all_scored_contours(frames: np.ndarray, all_scored: list):
    """
    Displays all contours per frame with their scores.
    Press any key to go to next frame. Press ESC to exit.
    """
    for idx, frame in enumerate(frames):
        display = frame.copy()
        contours_scores = all_scored[idx]

        for contour, score in contours_scores:
            cv2.drawContours(display, [contour], -1, (0, 255, 255), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(display, f"{score:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(display, f"Frame {idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("All Scored Contours", display)
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Combine detections to track ping-pong ball"
    )
    parser.add_argument('config', help='Path to JSON config file')
    parser.add_argument('--export-coords', dest='export_coords', help='Path to save coordinates JSON')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        full_cfg = json.load(f)

    npy_file = full_cfg['combine']['npy_file']
    #coords = TOP(npy_file, full_cfg)
    ''''''
    coords, all_scored = TOP_2(npy_file, full_cfg)
    frames = np.load(npy_file)
    display_all_scored_contours(frames, all_scored)
    ''''''
    for i, pt in enumerate(coords):
        print(f"Frame {i}: {pt}")

    if args.export_coords:
        with open(args.export_coords, 'w', encoding='utf-8') as of:
            json.dump({'coordinates': coords}, of, indent=4)
        print(f"Coordinates exported to {args.export_coords}")


def display_tracking(frames: np.ndarray, coords: list, radius: int = 10):
    """
    Display video frames one-by-one with the selected point drawn as a green circle.
    Press any key to advance to the next frame, or ESC to exit.

    :param frames: Array of frames (H, W, C)
    :param coords: List of (x, y) or None per frame
    :param radius: Radius of the drawn circle
    """
    for i, frame in enumerate(frames):
        display = frame.copy()
        pt = coords[i]
        if pt is not None:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(display, (x, y), radius, (0, 255, 0), 2)
        cv2.putText(display, f"Frame {i}/{len(frames)-1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)
        cv2.imshow("Tracking Viewer", display)
        key = cv2.waitKey(0)  # מחכה ללחיצת מקש
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
