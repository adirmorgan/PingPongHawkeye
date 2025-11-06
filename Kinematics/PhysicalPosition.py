import json
import argparse
import numpy as np
import cv2  # Replacing cv2.sfm import

from Combine import TOP_2D


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


def TOP_3D(frames_at_t: list[np.ndarray], full_cfg: dict) -> tuple[float, float, float] | None:
    """
    Triangulate a 3D point from multiple camera frames at a single time instant.

    Args:
        frames_at_t: list of single-frame arrays, one per camera
        full_cfg: loaded JSON config

    Returns:
        (x, y, z) or None
    """
    phys_cfg = full_cfg['PhysicalPosition']
    cameras_file = phys_cfg.get('cameras_file')
    P_list = build_projection_matrices(cameras_file)

    # Get corresponding 2D points from all camera frames
    pts2d = [TOP_2D(frame=frm, full_cfg=full_cfg) for frm in frames_at_t]

    # Validate that at least two 2D points were found
    if len(pts2d) < 2 or pts2d[0] is None or pts2d[1] is None:
        return None

    # Prepare 2D points for triangulation
    x1 = np.array(pts2d[0], dtype=float).reshape(2, 1)  # First camera
    x2 = np.array(pts2d[1], dtype=float).reshape(2, 1)  # Second camera

    # Perform triangulation using cv2.triangulatePoints
    X_hom = cv2.triangulatePoints(P_list[0], P_list[1], x1, x2)

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
    for i in range(n_frames):
        frames_at_t = [cam[i] for cam in all_frames]  # Extract frames for a single time instant
        point3d = TOP_3D(frames_at_t, full_cfg)
        entry = {'t': i / fps, 'x': None, 'y': None, 'z': None}
        if point3d:
            entry.update({'x': point3d[0], 'y': point3d[1], 'z': point3d[2]})
        trajectory.append(entry)

    with open(out_path, 'w') as f:
        json.dump(trajectory, f, indent=4)
    print(f"3D trajectory saved to {out_path}")





'''
''



''
'''




def run_gui(full_cfg):
    phys_cfg = full_cfg['PhysicalPosition']
    all_frames = [np.load(p) for p in phys_cfg['npy_files']]
    n_frames = all_frames[0].shape[0]

    paused = False
    idx = 0
    cv2.namedWindow('Physical Position GUI')
    while True:
        if not paused:
            idx = (idx + 1) % n_frames
        # combine 2D and draw
        out = []
        for cam_idx, frame in enumerate(all_frames):
            pt = TOP_2D(frame[idx], full_cfg)
            disp = frame[idx].copy()
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
            idx = (idx + 1) % n_frames
        elif paused and key == ord('a'):
            idx = (idx - 1) % n_frames
    cv2.destroyAllWindows()


def main_gui():
    parser = argparse.ArgumentParser(description="3D Trajectory GUI")
    parser.add_argument('config', help='Path to JSON config file')
    args = parser.parse_args()

    full_cfg = json.load(open(args.config,'r'))
    run_gui(full_cfg)

if __name__ == '__main__':
    main_gui()
