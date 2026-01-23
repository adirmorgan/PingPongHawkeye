import argparse
import concurrent.futures
from itertools import combinations

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

    pair_list = list(combinations(range(nc), 2))
    pts3d_flat = [pts3d[c1][c2] for (c1, c2) in pair_list]
    scr3d_flat = [scr3d[c1][c2] for (c1, c2) in pair_list]

    if nc == 2:
        return pts3d[0][1]

    if isinstance(method, (list, tuple)) and len(method) == 3 and method[0] == "select":
        cam1, cam2 = int(method[1]), int(method[2])
        return pts3d[cam1][cam2]

    if method == "majority":  # return the rounded result that got most votes
        pts_valid = [p for p in pts3d_flat if p is not None]
        if not pts_valid:
            return None
        arr = np.array(pts_valid, dtype=float)
        unique, counts = np.unique(arr, axis=0, return_counts=True)
        best = unique[np.argmax(counts)]
        return float(best[0]), float(best[1]), float(best[2])

    if method == "average":  # returns the (unweighted) average of all points
        pts_valid = [p for p in pts3d_flat if p is not None]
        if not pts_valid:
            return None
        arr = np.array(pts_valid, dtype=float)
        mean = arr.mean(axis=0)
        return float(mean[0]), float(mean[1]), float(mean[2])

    if method == "best":  # uses the point resulted by the pair with the best score
        best_score = None
        best_pt = None
        for pt, sc in zip(pts3d_flat, scr3d_flat):
            if pt is None:
                continue
            if best_score is None or sc > best_score:
                best_score = sc
                best_pt = pt
        return best_pt

    if method == "weighted":  # weighted average
        numer = np.zeros(3, dtype=float)
        denom = 0.0
        for pt, sc in zip(pts3d_flat, scr3d_flat):
            if pt is None or sc is None:
                continue
            sc = float(sc)
            if sc <= 0.0:
                continue
            numer += np.array(pt, dtype=float) * sc
            denom += sc
        if denom == 0.0:
            return None
        out = numer / denom
        return float(out[0]), float(out[1]), float(out[2])

    raise ValueError(f"Unknown merge method: {method}")


def TOP_3D(
    all_frames: list[np.ndarray],
    frame_index: int,
    full_cfg: dict,
    executor: concurrent.futures.Executor | None = None,
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

    # --- Parallelize TOP_2D across cameras (threads reused across frames if executor is passed)
    def _call_top2d(cam_idx: int):
        return TOP_2D(all_frames[cam_idx], frame_index, full_cfg=full_cfg)

    created_local_executor = False
    if executor is None:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, n_cameras)  # sensible default
        )
        created_local_executor = True

    try:
        res2d = list(executor.map(_call_top2d, range(n_cameras)))
    finally:
        if created_local_executor:
            executor.shutdown(wait=True)

    pts2d = [r[COORD_2D] for r in res2d]
    scr2d = [r[SCORE_2D] for r in res2d]

    if full_cfg.get("printing", False):
        printGrey(f"\t2D Results of Frame {frame_index}:")
        for cam in range(n_cameras):
            printGrey(f"\t\tCamera {cam} : {pts2d[cam]}")

    pair_list = list(combinations(range(n_cameras), 2))
    n_combs = len(pair_list)

    scr3d = [[0.0 for _ in range(n_cameras)] for _ in range(n_cameras)]
    pts3d = [[None for _ in range(n_cameras)] for _ in range(n_cameras)]

    for cam1, cam2 in pair_list:
        if pts2d[cam1] is None or pts2d[cam2] is None:
            continue

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

        X_hom /= X_hom[3, 0]

        pt3d = toleround(
            [float(X_hom[0, 0]), float(X_hom[1, 0]), float(X_hom[2, 0])],
            phys_cfg["tolerance"],
        )

        if np.linalg.norm(pt3d) > phys_cfg["max_distance"]:
            continue

        pts3d[cam1][cam2] = pt3d
        pts3d[cam2][cam1] = pt3d

    if full_cfg.get("printing", False):
        printGrey(f"\t3D Results of Frame {frame_index}:")

    raw_points = [None] * n_combs
    for i, (cam1, cam2) in enumerate(pair_list):
        raw_points[i] = pts3d[cam1][cam2]
        if full_cfg.get("printing", False):
            printGrey(f"\t\tCameras {cam1} & {cam2} : {raw_points[i]}")

    merge_method = phys_cfg.get("merge_method", None)
    if not merge_method:
        return None, raw_points

    merged = merge_predictions(merge_method, pts3d, scr3d, n_cameras)
    return merged, raw_points


@timeit("main (3D)")
def main():
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

    trajectory: list[dict] = []

    pair_trajectories = None
    if phys_cfg.get("save_pairs", False):
        pair_paths = phys_cfg.get("pairs_trajectories", [])
        if len(pair_paths) != n_combs:
            raise ValueError(
                f"pairs_trajectories has {len(pair_paths)} paths, but needs {n_combs} "
                f"(C(n_cameras,2))."
            )
        pair_trajectories = [[] for _ in range(n_combs)]

    # --- Create ONE thread pool for all frames (reused)
    max_workers = int(phys_cfg.get("top2d_workers", min(32, n_cameras)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for frame_idx in range(n_frames):
            with timeit(f"Frame {frame_idx} of {n_frames}"):
                pred_point, raw_points = TOP_3D(all_frames, frame_idx, full_cfg, executor=executor)
                t_now = frame_idx / fps

                if phys_cfg.get("merge_method", None):
                    entry = {"t": t_now, "x": None, "y": None, "z": None}
                    if pred_point is not None:
                        entry["x"] = float(pred_point[0])
                        entry["y"] = float(pred_point[1])
                        entry["z"] = float(pred_point[2])
                    trajectory.append(entry)

                if pair_trajectories is not None:
                    for i, pt in enumerate(raw_points):
                        entry = {"t": t_now, "x": None, "y": None, "z": None}
                        if pt is not None:
                            entry["x"] = float(pt[0])
                            entry["y"] = float(pt[1])
                            entry["z"] = float(pt[2])
                        pair_trajectories[i].append(entry)

    with timeit("Saving output trajectory"):
        with open(out_path, "w") as f:
            json.dump(trajectory, f, indent=4)

        if pair_trajectories is not None:
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


if __name__ == "__main__":
    main()
