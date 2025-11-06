#!/usr/bin/env python3
import json
import argparse
import numpy as np
import cv2

from BallDetection import MotionDetection, ShapeDetection, ColorDetection
from utils import get_coordinates, Contours


def TOP_2D(frame: np.ndarray, full_cfg: dict) -> tuple[int,int] | None:
    """
    Process a single frame and return the best-match centroid (x, y) or None.
    """
    combine_cfg = full_cfg['combine']
    shape_cfg   = full_cfg['shape_detection']
    color_cfg   = full_cfg['color_detection']
    motion_cfg  = full_cfg['motion_detection']

    s_w = combine_cfg['shape_weight']
    c_w = combine_cfg['color_weight']
    m_w = combine_cfg['motion_weight']
    min_score = combine_cfg.get('min_score', 0.0)

    best_score = min_score
    best_contour = None

    # wrap the single frame into a frames-array for Contours()
    frames = np.array([frame])
    contours = Contours(frames, 0, shape_cfg)

    for contour, _ in contours:
        s_score = ShapeDetection.Shape_Detection(frames, 0, contour, shape_cfg)
        c_score = ColorDetection.Color_Detection(frames, 0, contour, color_cfg)
        m_score = MotionDetection.Motion_Detection(frames, 0, contour, motion_cfg)

        strategy = combine_cfg.get('strategy', 'gather')
        match strategy:
            case "augment":   # weighted RMS
                score = (s_w*s_score**2 + c_w*c_score**2 + m_w*m_score**2) ** 0.5
            case "maximize":
                score = max(s_score, c_score, m_score)
            case "gather":    # weighted arithmetic mean
                score = (s_score * s_w + c_score * c_w + m_score * m_w)
            case "filter":    # weighted geometric mean
                total_w = s_w + c_w + m_w
                score = (s_score**s_w * c_score**c_w * m_score**m_w) ** (1.0/total_w)
            case "minimize":
                score = min(s_score, c_score, m_score)
            case "diminish":  # weighted harmonic mean
                denom = 0.0
                if s_score > 0: denom += s_w/s_score
                if c_score > 0: denom += c_w/c_score
                if m_score > 0: denom += m_w/m_score
                score = 1.0/denom if denom>0 else 0.0
            case _:
                score = 0.0

        if score > best_score:
            best_score   = score
            best_contour = contour

    return get_coordinates(best_contour) if best_contour is not None else None


def main():
    parser = argparse.ArgumentParser(
        description="Combine detections to track ping-pong ball (2D)"
    )
    parser.add_argument('config', help='Path to JSON config file')
    parser.add_argument(
        '--export-coords',
        dest='export_coords',
        help='Optional path to save 2D coordinates JSON'
    )
    args = parser.parse_args()

    # load full config
    with open(args.config, 'r', encoding='utf-8') as f:
        full_cfg = json.load(f)

    # load all frames
    npy_path = full_cfg['combine']['npy_file']
    frames = np.load(npy_path)

    coords = []
    for idx, frame in enumerate(frames):
        coord = TOP_2D(frame, full_cfg)
        coords.append(coord)
        if full_cfg['combine'].get('print', False):
            print(f"Frame {idx}: {coord}")

    if args.export_coords:
        with open(args.export_coords, 'w', encoding='utf-8') as of:
            json.dump({'coordinates': coords}, of, indent=4)
        print(f"2D coordinates exported to {args.export_coords}")


if __name__ == '__main__':
    main()
