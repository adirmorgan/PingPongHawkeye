import argparse

from BallDetection.Combine import TOP_2D
from utils import *  # assumes toleround, timeit, timing, etc.

CONTR_2D = 0  # index of field "contour" returned by TOP_2D
COORD_2D = 1  # index of field "coordinate" returned by TOP_2D
SCORE_2D = 2  # index of field "score" returned by TOP_2D


def build_projection_matrices(cameras_file: str) -> list[np.ndarray]:
    """
    Given camera parameters in cameras_file, build each camera's projection matrix P = K [R | t].
    """
    P_list = []
    cams = json.load(open(cameras_file, "r"))
    for cam in cams:
        K = np.array(cam["K"], dtype=float)
        rvec = np.array(cam["rvec"], dtype=float).reshape(3, 1)
        tvec = np.array(cam["tvec"], dtype=float).reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to rotation matrix
        Rt = np.hstack((R, tvec))   # Concatenate R and t (extrinsics)
        P = K @ Rt                  # Projection matrix
        P_list.append(P)
    return P_list


def merge_predictions(method, pts3d, scr3d, nc: int):
    """
    Args:
        method: merge method (string like "average"/"scores"/"weighted"/"majority"
                or list/tuple like ["select", cam1, cam2])
        pts3d: matrix [cam1][cam2] -> 3D point (x, y, z) or None
        scr3d: matrix [cam1][cam2] -> pair score (float)
        nc: total number of cameras

    Returns:
        (x, y, z) tuple or None
    """
    if nc < 2:
        raise ValueError("Not enough cameras to merge predictions.")

    # Precompute a stable ordering for pairs and flatten accordingly
    pair_list = list(combinations(range(nc), 2))
    pts3d_flat = [pts3d[c1][c2] for (c1, c2) in pair_list]
    scr3d_flat = [scr3d[c1][c2] for (c1, c2) in pair_list]

    # Handle the special case (2 cameras) naturally via below logic,
    # but keep this for clarity.
    if nc == 2:
        return pts3d[0][1]

    # "select" can be passed as ["select", cam1, cam2]
    if isinstance(method, (list, tuple)) and len(method) == 3 and method[0] == "select":
        cam1, cam2 = int(method[1]), int(method[2])
        return pts3d[cam1][cam2]

    if method == "majority":
        pts_valid = [p for p in pts3d_flat if p is not None]
        if not pts_valid:
            return None
        arr = np.array(pts_valid, dtype=float)  # shape (k,3)
        unique, counts = np.unique(arr, axis=0, return_counts=True)
        best = unique[np.argmax(counts)]
        return float(best[0]), float(best[1]), float(best[2])

    if method == "average":
        pts_valid = [p for p in pts3d_flat if p is not None]
        if not pts_valid:
            return None
        arr = np.array(pts_valid, dtype=float)
        mean = arr.mean(axis=0)
        return float(mean[0]), float(mean[1]), float(mean[2])

    if method == "scores":
        best_score = None
        best_pt = None
        for pt, sc in zip(pts3d_flat, scr3d_flat):
            if pt is None:
                continue
            if best_score is None or sc > best_score:
                best_score = sc
                best_pt = pt
        return best_pt

    if method == "weighted":
        num = np.zeros(3, dtype=float)
        den = 0.0
        for pt, sc in zip(pts3d_flat, scr3d_flat):
            if pt is None:
                continue
            if sc is None:
                continue
            sc = float(sc)
            if sc <= 0.0:
                continue
            num += np.array(pt, dtype=float) * sc
            den += sc
        if den == 0.0:
            return None
        out = num / den
        return float(out[0]), float(out[1]), float(out[2])

    raise ValueError(f"Unknown merge method: {method}")


def TOP_3D(
    all_frames: list[np.ndarray],
    frame_index: int,
    full_cfg: dict
) -> tuple[tuple[float, float, float] | None, list[tuple[float, float, float] | None]]:
    """
    Triangulate a 3D point from multiple camera frames at a single time instant.

    Returns:
        merged_point: (x, y, z) or None
        raw_points:   list of length C(n_cameras,2) with each entry (x,y,z) or None,
                      ordered by combinations(range(n_cameras),2)
    """
    phys_cfg = full_cfg["PhysicalPosition"]
    cameras_file = phys_cfg.get("cameras_file")
    P_list = build_projection_matrices(cameras_file)

    n_cameras = len(all_frames)
    if n_cameras < 2:
        print(f"Not enough cameras to triangulate. Found {n_cameras} cameras.")
        exit(1)

    # TOP_2D returns (contour, coordinate, score) for each camera
    res2d = [TOP_2D(all_frames[cam], frame_index, full_cfg=full_cfg) for cam in range(n_cameras)]
    pts2d = [r[COORD_2D] for r in res2d]  # 2D points (pixel indices)
    scr2d = [r[SCORE_2D] for r in res2d]  # 2D detection scores

    if full_cfg.get("printing", False):
        print(f"\t2D Results of Frame {frame_index}:")
        for cam in range(n_cameras):
            print(f"\t\tCamera {cam} : {pts2d[cam]}")

    # Pair list (stable ordering used everywhere)
    pair_list = list(combinations(range(n_cameras), 2))
    n_combs = len(pair_list)

    # Pair score: geometric mean of 2D scores (only meaningful if both detections exist)
    scr3d = [[0.0 for _ in range(n_cameras)] for _ in range(n_cameras)]
    pts3d = [[None for _ in range(n_cameras)] for _ in range(n_cameras)]

    for cam1, cam2 in pair_list:
        if pts2d[cam1] is None or pts2d[cam2] is None:
            continue

        # If your TOP_2D can return score None/0, you might want to guard here
        s1 = float(scr2d[cam1]) if scr2d[cam1] is not None else 0.0
        s2 = float(scr2d[cam2]) if scr2d[cam2] is not None else 0.0
        scr_pair = float(np.sqrt(max(s1, 0.0) * max(s2, 0.0)))
        scr3d[cam1][cam2] = scr_pair
        scr3d[cam2][cam1] = scr_pair

        x1 = np.array(pts2d[cam1], dtype=float).reshape(2, 1)
        x2 = np.array(pts2d[cam2], dtype=float).reshape(2, 1)

        X_hom = cv2.triangulatePoints(P_list[cam1], P_list[cam2], x1, x2)
        if X_hom.shape[0] != 4 or X_hom[3, 0] == 0:
            continue

        X_hom /= X_hom[3, 0]  # normalize

        pt3d = toleround(
            [float(X_hom[0, 0]), float(X_hom[1, 0]), float(X_hom[2, 0])],
            phys_cfg["tolerance"],
        )

        # Distance gating
        if np.linalg.norm(pt3d) > phys_cfg["max_distance"]:
            continue

        pts3d[cam1][cam2] = pt3d
        pts3d[cam2][cam1] = pt3d

    if full_cfg.get("printing", False):
        print(f"\t3D Results of Frame {frame_index}:")

    # Flatten raw points in stable order
    raw_points = [None] * n_combs
    for i, (cam1, cam2) in enumerate(pair_list):
        raw_points[i] = pts3d[cam1][cam2]
        if full_cfg.get("printing", False):
            print(f"\t\tCameras {cam1} & {cam2} : {raw_points[i]}")

    merge_method = phys_cfg.get("merge_method", None)
    if not merge_method:
        return None, raw_points

    merged = merge_predictions(merge_method, pts3d, scr3d, n_cameras)
    return merged, raw_points


@timeit("main (3D)")
def main_auto():
    parser = argparse.ArgumentParser(
        description="Compute 3D trajectory by looping over frames and triangulating per-instant"
    )
    parser.add_argument("config", help="Path to JSON config file")
    args = parser.parse_args()

    full_cfg = json.load(open(args.config, "r"))
    phys_cfg = full_cfg["PhysicalPosition"]
    timing(full_cfg["timing"])

    npy_files = phys_cfg["npy_files"]
    fps = phys_cfg["frame_rate"]
    out_path = phys_cfg["output_trajectory"]

    all_frames = [np.load(path) for path in npy_files]
    n_cameras = len(all_frames)
    n_frames = all_frames[0].shape[0]

    pair_list = list(combinations(range(n_cameras), 2))
    n_combs = len(pair_list)

    # --- Merged trajectory output (list of dicts)
    trajectory: list[dict] = []

    # --- Pair trajectories output (list per pair, each is list of dicts like coordinates.json)
    pair_trajectories = None
    if phys_cfg.get("save_pairs", False):
        pair_paths = phys_cfg.get("pairs_trajectories", [])
        if len(pair_paths) != n_combs:
            raise ValueError(
                f"pairs_trajectories has {len(pair_paths)} paths, but needs {n_combs} "
                f"(C(n_cameras,2))."
            )
        pair_trajectories = [[] for _ in range(n_combs)]

    for frame_idx in range(n_frames):
        with timeit(f"Frame {frame_idx} of {n_frames}"):
            pred_point, raw_points = TOP_3D(all_frames, frame_idx, full_cfg)
            t_now = frame_idx / fps

            # Save merged output only if merge_method is enabled
            if phys_cfg.get("merge_method", None):
                entry = {"t": t_now, "x": None, "y": None, "z": None}
                if pred_point is not None:
                    entry["x"] = float(pred_point[0])
                    entry["y"] = float(pred_point[1])
                    entry["z"] = float(pred_point[2])
                trajectory.append(entry)

            # Save each pair in the SAME schema: list of dicts with t,x,y,z (None => null)
            if pair_trajectories is not None:
                for i, pt in enumerate(raw_points):
                    e = {"t": t_now, "x": None, "y": None, "z": None} # entries' stracture
                    if pt is not None:
                        e["x"] = float(pt[0])
                        e["y"] = float(pt[1])
                        e["z"] = float(pt[2])
                    pair_trajectories[i].append(e)

    with timeit("Saving output trajectory"):
        with open(out_path, "w") as f:
            json.dump(trajectory, f, indent=4)

        if pair_trajectories is not None:
            # Optional metadata mapping: which index corresponds to which (cam1, cam2)
            meta_path = phys_cfg.get("pairs_meta_path", None)
            if meta_path:
                meta = [{"pair_index": i, "cam1": int(c1), "cam2": int(c2)}
                        for i, (c1, c2) in enumerate(pair_list)]
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=4)

            for i in range(n_combs):
                pair_path = phys_cfg["pairs_trajectories"][i]
                with open(pair_path, "w") as f:
                    json.dump(pair_trajectories[i], f, indent=4)

    print(f"3D trajectory saved to {out_path}")


def run_gui(full_cfg):
    phys_cfg = full_cfg["PhysicalPosition"]
    all_frames = [np.load(p) for p in phys_cfg["npy_files"]]
    n_frames = all_frames[0].shape[0]

    paused = False
    cv2.namedWindow("Physical Position GUI")
    for frame_idx in range(n_frames):
        if paused:
            frame_idx = frame_idx - 1

        out = []
        for cam_idx in range(len(all_frames)):
            frames = all_frames[cam_idx]
            pt = TOP_2D(frames, frame_idx, full_cfg)
            disp = frames[frame_idx].copy()
            if pt is not None:
                # pt from TOP_2D likely contains coordinate at index COORD_2D; adjust if needed
                # Here you used pt[0],pt[1] earlier; keep as-is.
                cv2.rectangle(disp, (pt[0] - 5, pt[1] - 5), (pt[0] + 5, pt[1] + 5), (0, 0, 255), 2)
            out.append(disp)

        concat = np.hstack(out)
        cv2.imshow("Physical Position GUI", concat)
        key = cv2.waitKey(phys_cfg.get("display_fps_delay", 30)) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif paused and key == ord("d"):
            frame_idx = (frame_idx + 1) % n_frames
        elif paused and key == ord("a"):
            frame_idx = (frame_idx - 1) % n_frames
        if key == ord("w"):
            frame_idx = (frame_idx + 10) % n_frames
        if key == ord("s"):
            frame_idx = (frame_idx - 10) % n_frames

    cv2.destroyAllWindows()


def main_gui():
    parser = argparse.ArgumentParser(description="3D Trajectory GUI")
    parser.add_argument("config", help="Path to JSON config file")
    args = parser.parse_args()

    full_cfg = json.load(open(args.config, "r"))
    timing(full_cfg["timing"])
    run_gui(full_cfg)


if __name__ == "__main__":
    choice = input("GUI or AUTO? (G/A) ")
    if choice in ("G", "g"):
        main_gui()
    elif choice in ("A", "a"):
        main_auto()
    else:
        print("Invalid choice. Exiting.")
