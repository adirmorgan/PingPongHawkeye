import json
import argparse
import numpy as np
import cv2

from BallDetection import MotionDetection, ShapeDetection, ColorDetection
from utils import get_coordinates, Contours


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
    coords = TOP(npy_file, full_cfg)

    for i, pt in enumerate(coords):
        print(f"Frame {i}: {pt}")

    if args.export_coords:
        with open(args.export_coords, 'w', encoding='utf-8') as of:
            json.dump({'coordinates': coords}, of, indent=4)
        print(f"Coordinates exported to {args.export_coords}")

if __name__ == '__main__':
    main()
