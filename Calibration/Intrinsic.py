# intrinsic_calibration_gui.py
import json
import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict, List


def load_camera_db(json_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if raw == "":
        return []
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "cameras" in data and isinstance(data["cameras"], list):
        return data["cameras"]
    raise RuntimeError("Camera JSON file must contain either a list or a dict with a 'cameras' list.")


def save_camera_db(json_path: str, cameras: List[Dict[str, Any]]) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cameras, f, indent=4)


def find_camera_entry(cameras: List[Dict[str, Any]], camera_name: str) -> Optional[Dict[str, Any]]:
    for cam in cameras:
        if cam.get("name") == camera_name:
            return cam
    return None


def ndarray_to_K_list(K: np.ndarray) -> List[List[float]]:
    return [[float(K[r, c]) for c in range(3)] for r in range(3)]


def ndarray_to_tvec_list(tvec: np.ndarray) -> List[float]:
    arr = np.asarray(tvec, dtype=float).reshape(-1)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def ndarray_to_rvec_row_list(rvec: np.ndarray) -> List[List[float]]:
    arr = np.asarray(rvec, dtype=float).reshape(-1)
    return [[float(arr[0]), float(arr[1]), float(arr[2])]]


def print_camera_entry(entry: Dict[str, Any]) -> None:
    print(json.dumps(entry, indent=4))


@dataclass
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int


class IntrinsicCalibrationGUI:
    def __init__(self, json_path: str) -> None:
        self.json_path = json_path
        self.root = tk.Tk()
        self.root.withdraw()

        self.camera_name: str = "camera"
        self.pattern_cols: int = 9
        self.pattern_rows: int = 6
        self.square_size: float = 25.0
        self.chessboard_video: Optional[VideoMetadata] = None

        self.K: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.rms_error: Optional[float] = None

    @staticmethod
    def print_progress(message: str) -> None:
        gray = "\033[90m"
        reset = "\033[0m"
        print(f"{gray}{message}{reset}")

    @staticmethod
    def read_video_metadata(path: str) -> VideoMetadata:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return VideoMetadata(path=path, width=width, height=height, fps=fps, frame_count=frame_count)

    def ask_inputs(self) -> bool:
        self.camera_name = simpledialog.askstring(
            "Camera Name",
            "Enter camera name exactly as you want it stored:",
            initialvalue="camera"
        ) or "camera"

        pattern_text = simpledialog.askstring(
            "Chessboard Pattern Size",
            "Enter number of INNER corners as cols,rows\nExample: 9,6",
            initialvalue="9,6",
        )
        if pattern_text is None:
            return False
        try:
            cols_text, rows_text = pattern_text.replace(" ", "").split(",")
            self.pattern_cols = int(cols_text)
            self.pattern_rows = int(rows_text)
        except Exception:
            messagebox.showerror("Invalid Input", "Pattern size must be entered as cols,rows")
            return False

        square_size = simpledialog.askfloat(
            "Square Size",
            "Enter chessboard square size in real-world units (e.g. mm):",
            initialvalue=25.0,
            minvalue=1e-9,
        )
        if square_size is None:
            return False
        self.square_size = float(square_size)

        chessboard_path = filedialog.askopenfilename(
            title="Select chessboard video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV"), ("All files", "*.*")],
        )
        if not chessboard_path:
            return False

        self.print_progress("[Intrinsic] Opening chessboard video...")
        self.chessboard_video = self.read_video_metadata(chessboard_path)
        self.print_progress(
            f"[Intrinsic] Video metadata: {self.chessboard_video.width}x{self.chessboard_video.height}, "
            f"fps={self.chessboard_video.fps:.3f}, frames={self.chessboard_video.frame_count}"
        )
        self.print_progress(f"[Intrinsic] Chessboard pattern: {self.pattern_cols}x{self.pattern_rows} inner corners")
        self.print_progress(f"[Intrinsic] Square size: {self.square_size}")
        return True

    def calibrate(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.chessboard_video is None:
            raise RuntimeError("Chessboard video was not loaded.")

        cap = cv2.VideoCapture(self.chessboard_video.path)
        if not cap.isOpened():
            raise RuntimeError("Could not reopen chessboard video.")

        pattern_size = (self.pattern_cols, self.pattern_rows)
        objp = np.zeros((self.pattern_rows * self.pattern_cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_cols, 0:self.pattern_rows].T.reshape(-1, 2)
        objp *= self.square_size

        object_points = []
        image_points = []
        used_frames = 0
        frame_size = None
        frame_idx = 0
        checked_index = 0
        last_percent_printed = -1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        sample_every = max(1, self.chessboard_video.frame_count // 120)
        estimated_checked = max(1, (self.chessboard_video.frame_count + sample_every - 1) // sample_every)
        self.print_progress(f"[Intrinsic] Sampling approximately every {sample_every} frame(s)...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every != 0:
                frame_idx += 1
                continue

            checked_index += 1
            progress_percent = int(round(100.0 * checked_index / estimated_checked))
            if progress_percent >= last_percent_printed + 10:
                self.print_progress(
                    f"[Intrinsic] Scanning video... {min(progress_percent, 100)}% | "
                    f"checked={checked_index}/{estimated_checked} | detections={used_frames}"
                )
                last_percent_printed = progress_percent

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray,
                pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
            )

            if found:
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                object_points.append(objp.copy())
                image_points.append(corners_refined)
                frame_size = (gray.shape[1], gray.shape[0])
                used_frames += 1
                self.print_progress(
                    f"[Intrinsic] Chessboard detected in sampled frame around index {frame_idx}. "
                    f"Total valid detections: {used_frames}"
                )

            frame_idx += 1

        cap.release()
        self.print_progress(f"[Intrinsic] Finished scanning. Total valid detections: {used_frames}")

        if used_frames < 4:
            raise RuntimeError(f"Only {used_frames} valid chessboard detections were found. Need at least 4 good views.")

        self.print_progress("[Intrinsic] Running cv2.calibrateCamera()...")
        rms_error, K, dist_coeffs, _, _ = cv2.calibrateCamera(
            object_points,
            image_points,
            frame_size,
            None,
            None,
        )

        self.K = K
        self.dist_coeffs = dist_coeffs
        self.rms_error = float(rms_error)
        self.print_progress(f"[Intrinsic] Calibration complete. RMS reprojection error = {self.rms_error:.6f}")
        return K, dist_coeffs

    def update_json_file(self) -> Dict[str, Any]:
        if self.K is None:
            raise RuntimeError("K is not available.")
        cameras = load_camera_db(self.json_path)
        entry = find_camera_entry(cameras, self.camera_name)
        if entry is None:
            entry = {"name": self.camera_name}
            cameras.append(entry)
        entry["K"] = ndarray_to_K_list(self.K)
        save_camera_db(self.json_path, cameras)
        return entry

    def run(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        try:
            if not self.ask_inputs():
                return None, None, None
            K, dist_coeffs = self.calibrate()
            self.update_json_file()
            return K, dist_coeffs, self.camera_name
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            print(f"Error: {exc}")
            return None, None, None
        finally:
            self.root.quit()
            self.root.destroy()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python intrinsic_calibration_gui.py <camera_json_path>")
    app = IntrinsicCalibrationGUI(sys.argv[1])
    app.run()


