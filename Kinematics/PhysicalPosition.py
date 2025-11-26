import json
import argparse
from itertools import combinations

import numpy as np
import cv2  # Replacing cv2.sfm import

from Combine import TOP_2D
from utils import *

COORD_2D = 0 # index of field "coordinate" returned by TOP_2D
SCORE_2D = 1 # index of field "score" returned by TOP_2D


def build_projection_matrices(cameras_file):
    """
    Builds P = K[R|t] handling both Position conversion AND Coordinate System conversion.
    """
    P_list = []
    cams = json.load(open(cameras_file, 'r'))

    # --- מטריצת התיקון (Magic Matrix) ---
    # הופכת את Y ואת Z.
    # נדרש כמעט תמיד במעבר מ-Unity/Blender ל-OpenCV
    fix_axes = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=float)

    for i, cam in enumerate(cams):
        K = np.array(cam['K'], dtype=float)
        rvec = np.array(cam['rvec'], dtype=float).reshape(3, 1)
        C = np.array(cam['tvec'], dtype=float).reshape(3, 1)  # Camera Position

        # 1. המרת וקטור סיבוב למטריצה
        R_sim, _ = cv2.Rodrigues(rvec)

        # 2. --- התיקון החדש: התאמת מערכת צירים ---
        # אנחנו מכפילים מימין כדי לסובב את מערכת הצירים של המצלמה עצמה
        R_cv = R_sim @ fix_axes

        # 3. חישוב וקטור ההזזה (שימוש ב-R המתוקן!)
        t_vec = -R_cv @ C

        # 4. בניית P
        Rt = np.hstack((R_cv, t_vec))
        P = K @ Rt

        P_list.append(P)

        # הדפסת בדיקה
        verify_camera_center(P, C, i)

    return P_list


def verify_camera_center(P, C_original, cam_index):
    C_hom = np.append(C_original, [[1]], axis=0)
    projected = P @ C_hom
    error = np.linalg.norm(projected)
    if error > 1e-3:
        print(f"⚠️ Cam {cam_index}: Center projection error is {error:.5f} (Should be ~0)")

def TOP_3D(all_frames: list[np.ndarray], frame_index:int ,full_cfg: dict) -> tuple[float, float, float] | None:
    """
    Triangulate a 3D point from multiple camera frames at a single time instant.

    Args:
        all_frames: list of all frames from all cameras
        frame_index: index of the frame to triangulate
        full_cfg: loaded JSON config

    Returns:
        (x, y, z) or None
    """
    # DEBUG:
    print(f"frame index: {frame_index}")
    # ENDEBUG
    phys_cfg = full_cfg['PhysicalPosition']
    cameras_file = phys_cfg.get('cameras_file')
    P_list = build_projection_matrices(cameras_file)
    n_cameras = len(all_frames)

    if n_cameras < 2:
        print(f"Not enough cameras to triangulate. Found {n_cameras} cameras.")
        exit(1)

    # Get corresponding 2D points and scores from all camera frames
    '''NOTE:  TOP_2D returns a list of tuples (coordinate, score) for each camera.'''

    res2d = [TOP_2D(all_frames[cam], frame_index, full_cfg=full_cfg) for cam in range(len(all_frames))]

    pts2d = [r[COORD_2D] for r in res2d] # point coordinates in  2D (pixel indecies)
    scr2d =  [r[SCORE_2D] for r in res2d] # score of 2D detection
    # Get corresponding 3D point of all pairs (triangulation in pairs)
    pts3d = [[None for _ in range(n_cameras)] for _ in range(n_cameras)]
    for cam1, cam2 in combinations(range(n_cameras), 2):
        # Validate that at least two 2D points were found
        if len(pts2d) < 2 or pts2d[cam1] is None or pts2d[cam2] is None:
            continue
    
        # Prepare 2D points for triangulation
        x1 = np.array(pts2d[cam1], dtype=float).reshape(2, 1)  # First camera
        x2 = np.array(pts2d[cam2], dtype=float).reshape(2, 1)  # Second camera
    
        # Perform triangulation using cv2.triangulatePoints
        X_hom = cv2.triangulatePoints(P_list[cam1], P_list[cam2], x1, x2)
    
        # Convert homogeneous coordinates to 3D coordinates
        X_hom /= X_hom[3, 0] # Normalize
        # Build the symetric matrix of triangulation results.
        pt3d = toleround(
            [float(X_hom[0, 0]), float(X_hom[1, 0]), float(X_hom[2, 0])],
            phys_cfg['tolerance']
        )
        # TODO : (optional) this kind of results could be filtered earlier on 2D level...
        if np.linalg.norm(pt3d) > phys_cfg['max_distance']:
            continue # result will be None
        pts3d[cam1][cam2] = pt3d
        pts3d[cam2][cam1] = pt3d
    # TODO: split into functions and call each one if relevant.
    match phys_cfg['merge_method']:
        case ["select", cam1, cam2]: # no merging, pre-selected cameras
            return pts3d[int(cam1)][int(cam2)]
        case "majority": # merge according to majority of votes after rounding
            pts3d_flat = [
                pts3d[cam1][cam2]
                for cam1, cam2 in combinations(range(n_cameras), 2)
                if pts3d[cam1][cam2] is not None
            ]
            if not pts3d_flat:
                return None  # no results were found
            unique, counts = np.unique(pts3d_flat, axis=0, return_counts=True)
            max_idx = np.argmax(counts)
            best = unique[max_idx]
            # return tuple of floats.
            return float(best[0]), float(best[1]), float(best[2])
        case "average":
            return None #TODO : implement this merging method
        case "score":
            return None #TODO : implement this merging method
        case _: # do not merge, just print all trajectories simult.
            for c1 in range(n_cameras):
                for c2 in range(c1 + 1, n_cameras):
                    if pts3d[c1][c2] is not None:
                        print(f"Frame {frame_index}: Cam {c1}&{c2} -> {pts3d[c1][c2]}")
            return None




def main():
    parser = argparse.ArgumentParser(
        description="Compute 3D trajectory by looping over frames and triangulating per-instant"
    )
    parser.add_argument('config', help='Path to JSON config file')
    args = parser.parse_args()

    full_cfg = json.load(open(args.config, 'r'))
    phys_cfg = full_cfg['PhysicalPosition']
    timing(full_cfg['timing'])
    npy_files = phys_cfg['npy_files']
    fps = phys_cfg['frame_rate']
    out_path = phys_cfg['output_trajectory']

    # Load camera frames
    all_frames = [np.load(path) for path in npy_files]
    n_frames = all_frames[0].shape[0]

    trajectory = []
    for frame_idx in range(n_frames):
        with timeit(f"Frame {frame_idx} of {n_frames}"):
            point3d = TOP_3D(all_frames, frame_idx, full_cfg)
            entry = {'t': frame_idx / fps, 'x': None, 'y': None, 'z': None}
            if point3d is not None:
                x, y, z = point3d
                entry.update({'x': float(x), 'y': float(y), 'z': float(z)})
            trajectory.append(entry)

    with open(out_path, 'w') as f:
        json.dump(trajectory, f, indent=4)
    print(f"3D trajectory saved to {out_path}")

def run_gui(full_cfg):
    phys_cfg = full_cfg['PhysicalPosition']
    all_frames = [np.load(p) for p in phys_cfg['npy_files']]
    n_frames = all_frames[0].shape[0]

    paused = False
    cv2.namedWindow('Physical Position GUI')
    for frame_idx in range(n_frames):
        if paused:
            frame_idx = frame_idx - 1
        # combine 2D and draw
        out = []
        for cam_idx in range(len(all_frames)):
            frames = all_frames[cam_idx]
            pt,_ = TOP_2D(frames, frame_idx, full_cfg)
            disp = frames[frame_idx].copy()
            if pt is not None:
                cv2.rectangle(disp, (pt[0]-5,pt[1]-5),(pt[0]+5,pt[1]+5),(0,0,255),2)
            out.append(disp)
        # stack horizontally
        concat = np.hstack(out)
        cv2.imshow('Physical Position GUI', concat)
        key = cv2.waitKey(phys_cfg.get('display_fps_delay',30)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif paused and key == ord('d'):
           frame_idx = (frame_idx + 1) % n_frames
        elif paused and key == ord('a'):
           frame_idx = (frame_idx - 1) % n_frames
        if key == ord('w'):
           frame_idx = (frame_idx + 10) % n_frames
        if key == ord('s'):
           frame_idx = (frame_idx - 10) % n_frames


    cv2.destroyAllWindows()


def main_gui():
    parser = argparse.ArgumentParser(description="3D Trajectory GUI")
    parser.add_argument('config', help='Path to JSON config file')
    args = parser.parse_args()

    full_cfg = json.load(open(args.config,'r'))
    timing(full_cfg['timing'])
    run_gui(full_cfg)

if __name__ == '__main__':
    choice = input("GUI or AUTO? (G/A) ")
    if choice == 'G' or choice == 'g':
        main_gui()
    elif choice == 'A' or choice == 'a':
        main()
    else:
        print("Invalid choice. Exiting.")
