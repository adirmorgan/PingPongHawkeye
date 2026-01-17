import argparse
from itertools import combinations

from BallDetection.Combine import TOP_2D
from utils import *

CONTR_2D = 0 # index of field "contour" returned by TOP_2D
COORD_2D = 1 # index of field "coordinate" returned by TOP_2D
SCORE_2D = 2 # index of field "score" returned by TOP_2D

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


def merge_predictions(method: str, pts3d, scr3d, nc :int) -> tuple[float, float, float] | None:
    '''
    args:
        method: method used for merging predictions
        pts3d: mapping from [cam1][cam2] to the 3D point (x, y, z) of the triangulated by the pair (cam1, cam2)
        scr3d: mapping from [cam1][cam2] to the sum of scores of the pair (cam1, cam2)
        nc: number of cameras in total
    return:
        a single 3D point (x, y, z) representing final merged prediction. None if no predictions were found.
    '''

    if (nc < 2) : raise ValueError("Not enough cameras to merge predictions.")
    if (nc == 2) : return pts3d[0][1]
    '''
    NOTE: it is easier to use a vector (pair <-> idx) rather than a matrix (pair <-> idx1, idx2)
    '''
    scr3d_flat = [scr3d[cam1][cam2] for cam1, cam2 in combinations(range(nc), 2)]
    pts3d_flat = [pts3d[cam1][cam2] for cam1, cam2 in combinations(range(nc), 2)]

    match method:
        case ["select", cam1, cam2]: # no merging, pre-selected cameras
            return pts3d[int(cam1)][int(cam2)]
        case "majority": # merge according to majority of votes after rounding
            if not pts3d_flat:
                return None  # no results were found
            unique, counts = np.unique(pts3d_flat, axis=0, return_counts=True)
            max_idx = np.argmax(counts)
            best = unique[max_idx]
            # return tuple of floats.
            return float(best[0]), float(best[1]), float(best[2])
        case "average": # average all results
            sum_pts = np.array(pts3d_flat).sum(axis=0)
            return sum_pts / len(pts3d_flat)
        case "scores": # choose the pair with the highest sum of scores
            best_score = 0
            for pair, score in enumerate(scr3d_flat):
                 if score > best_score:
                     best_score = score
                     best_pair = pair
            return pts3d_flat[best_pair]
        case "weighted":
            sum_pts = np.array(pts3d_flat) * np.array(scr3d_flat)[:, None]
            sum_scr = np.array(scr3d_flat).sum()
            return sum_pts / sum_scr
        case _:
            raise ValueError(f"Unknown merge method: {method}")

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
    n_cameras = len(all_frames)

    if n_cameras < 2:
        print(f"Not enough cameras to triangulate. Found {n_cameras} cameras.")
        exit(1)

    # Get corresponding 2D points and scores from all camera frames
    '''NOTE:  TOP_2D returns a list of tuples (coordinate, score) for each camera.'''

    res2d = [TOP_2D(all_frames[cam], frame_index, full_cfg=full_cfg) for cam in range(len(all_frames))]

    pts2d = [r[COORD_2D] for r in res2d] # point coordinates in  2D (pixel indecies)
    scr2d =  [r[SCORE_2D] for r in res2d] # score of 2D detection

    if (full_cfg['printing']):
        print(f"\t2D Results of Frame {frame_index}:")
        for cam in range(n_cameras):
            print(f"\t\tCamera {cam} : {pts2d[cam]}")
    # Get corresponding 3D point and scores of all pairs (triangulation in pairs)
    scr3d = [[np.sqrt(scr2d[cam1] * scr2d[cam2]) for cam1 in range(n_cameras)] for cam2 in range(n_cameras)] # score of a pair is the geometric mean
    pts3d = [[None for _ in range(n_cameras)] for _ in range(n_cameras)]
    for cam1, cam2 in combinations(range(n_cameras), 2):
        # Valindate that at least two 2D points were found
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
        if np.linalg.norm(pt3d) > phys_cfg['max_distance']: # TODO : (optional) this kind of results could be filtered earlier on 2D level...

            continue # result will be None
        pts3d[cam1][cam2] = pt3d
        pts3d[cam2][cam1] = pt3d

    if (not phys_cfg['merge_method'] or full_cfg['printing']): # do not merge, just print all coordinates as you go
        print(f"\t3D Results of Frame {frame_index}:")
        for cam1, cam2 in combinations(range(n_cameras), 2):
            print(f"\t\tCameras {cam1} & {cam2} : {pts3d[cam1][cam2]}")
        if not phys_cfg['merge_method'] : return None

    return merge_predictions(phys_cfg['merge_method'], pts3d, scr3d, n_cameras)

@ timeit("main (3D)")
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
    shape_frame = print(all_frames[0].shape)  # DEBUG

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
            pt = TOP_2D(frames, frame_idx, full_cfg)
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
    timing(full_cfg['timing']) # enable/disable timing prints while running
    run_gui(full_cfg)

if __name__ == '__main__':
    choice = input("GUI or AUTO? (G/A) ")
    if choice == 'G' or choice == 'g':
        main_gui()
    elif choice == 'A' or choice == 'a':
        main()
    else:
        print("Invalid choice. Exiting.")
