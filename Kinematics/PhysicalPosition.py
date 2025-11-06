import json
import argparse
import numpy as np
import cv2  # Replacing cv2.sfm import

from Combine import TOP_2D
from utils import *

def build_projection_matrices(cameras_file):
    """
    Given camera parameters in cameras_file, build each camera's projection matrix P = K [R | t].
    """
    P_list = []
    cams = json.load(open(cameras_file, 'r'))
    for cam in cams:
        K = np.array(cam['K'], dtype=float)
        rvec = np.array(cam['rvec'], dtype=float).reshape(3, 1)
        tvec = np.array(cam['tvec'], dtype=float).reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to rotation matrix
        Rt = np.hstack((R, tvec))  # Concatenate R and t (extrinsics)
        P = K @ Rt  # Projection matrix
        P_list.append(P)
    return P_list

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
    phys_cfg = full_cfg['PhysicalPosition']
    cameras_file = phys_cfg.get('cameras_file')
    P_list = build_projection_matrices(cameras_file)

    # Get corresponding 2D points from all camera frames
    pts2d = [TOP_2D(frames, frame_index, full_cfg=full_cfg) for frames in all_frames]

    # TODO: choose cameras-pair wisely (sensor merging)
    cam1 = 0  # arbitrary choice, not a wise one...
    cam2 = 1  # arbitrary choice, not a wise one...

    # Validate that at least two 2D points were found
    if len(pts2d) < 2 or pts2d[cam1] is None or pts2d[cam2] is None:
        return None

    # Prepare 2D points for triangulation
    x1 = np.array(pts2d[cam1], dtype=float).reshape(2, 1)  # First camera
    x2 = np.array(pts2d[cam2], dtype=float).reshape(2, 1)  # Second camera

    # Perform triangulation using cv2.triangulatePoints
    X_hom = cv2.triangulatePoints(P_list[cam1], P_list[cam2], x1, x2)

    # Convert homogeneous coordinates to 3D coordinates
    X_hom /= X_hom[3, 0]
    return float(X_hom[0, 0]), float(X_hom[1, 0]), float(X_hom[2, 0])


def main():
    parser = argparse.ArgumentParser(
        description="Compute 3D trajectory by looping over frames and triangulating per-instant"
    )
    parser.add_argument('config', help='Path to JSON config file')
    args = parser.parse_args()

    full_cfg = json.load(open(args.config, 'r'))
    phys_cfg = full_cfg['PhysicalPosition']
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
            if point3d:
                entry.update({'x': point3d[0], 'y': point3d[1], 'z': point3d[2]})
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
            pt = TOP_2D(frames, frame_idx, full_cfg)
            disp = frames[frame_idx].copy()
            if pt:
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
    run_gui(full_cfg)

if __name__ == '__main__':
    choice = input("GUI or AUTO? (G/A) ")
    if choice == 'G' or choice == 'g':
        main_gui()
    elif choice == 'A' or choice == 'a':
        main()
    else:
        print("Invalid choice. Exiting.")
