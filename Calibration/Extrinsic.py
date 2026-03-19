# extrinsic_calibration_gui.py
from Intrinsic import IntrinsicCalibrationGUI
import json
import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict


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


def K_list_to_ndarray(K_list: Any) -> np.ndarray:
    K = np.array(K_list, dtype=np.float64)
    if K.shape != (3, 3):
        raise RuntimeError("Stored K must have shape 3x3.")
    return K


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


class ExtrinsicCalibrationGUI:
    def __init__(self, json_path: str) -> None:
        self.json_path = json_path
        self.root = tk.Tk()
        self.root.withdraw()

        self.camera_name: str = "camera"
        self.K: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = np.zeros((5, 1), dtype=np.float64)

        self.pnp_video: Optional[VideoMetadata] = None
        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None

        self.selected_points_2d: List[Tuple[float, float]] = []
        self.selected_points_3d: List[Tuple[float, float, float]] = []
        self.current_frame: Optional[np.ndarray] = None
        self.current_display: Optional[np.ndarray] = None
        self.frame_index: int = 0
        self.cap_pnp: Optional[cv2.VideoCapture] = None
        self.zoom: float = 1.0
        self.zoom_step: float = 1.25
        self.min_zoom: float = 0.25
        self.max_zoom: float = 8.0
        self.margin_px: int = 220

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

    def resolve_camera_name_and_intrinsics(self) -> bool:
        self.camera_name = simpledialog.askstring(
            "Camera Name",
            "Enter camera name:",
            initialvalue="camera",
        ) or "camera"

        cameras = load_camera_db(self.json_path)
        entry = find_camera_entry(cameras, self.camera_name)

        if entry is not None and entry.get("K") is not None:
            self.K = K_list_to_ndarray(entry["K"])
            return True

        messagebox.showinfo(
            "Intrinsic Parameters Required",
            "This camera does not have a stored K matrix yet. The intrinsic calibration GUI will open now."
        )

        intrinsic_app = IntrinsicCalibrationGUI(self.json_path)
        K, _, camera_name = intrinsic_app.run()
        if K is None:
            messagebox.showerror("Missing Intrinsics", "Extrinsic calibration cannot continue without K.")
            return False

        self.camera_name = camera_name or self.camera_name
        cameras = load_camera_db(self.json_path)
        entry = find_camera_entry(cameras, self.camera_name)
        if entry is None or entry.get("K") is None:
            messagebox.showerror("Missing Intrinsics", "K was not found in the JSON file after intrinsic calibration.")
            return False
        self.K = K_list_to_ndarray(entry["K"])
        self.root = tk.Tk()
        self.root.withdraw()
        return True

    def ask_inputs(self) -> bool:
        pnp_path = filedialog.askopenfilename(
            title="Select scene video for PnP extrinsic calibration",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV"), ("All files", "*.*")],
        )
        if not pnp_path:
            return False

        self.pnp_video = self.read_video_metadata(pnp_path)
        return True

    def load_pnp_video(self) -> None:
        if self.pnp_video is None:
            raise RuntimeError("PnP video was not loaded.")
        self.cap_pnp = cv2.VideoCapture(self.pnp_video.path)
        if not self.cap_pnp.isOpened():
            raise RuntimeError("Could not open PnP video.")
        self.frame_index = 0
        self.set_frame(self.frame_index)

    def set_frame(self, index: int) -> None:
        if self.cap_pnp is None or self.pnp_video is None:
            raise RuntimeError("PnP video capture is not initialized.")
        index = max(0, min(index, self.pnp_video.frame_count - 1))
        self.cap_pnp.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap_pnp.read()
        if not ret:
            raise RuntimeError(f"Could not read frame {index} from PnP video.")
        self.frame_index = index
        self.current_frame = frame
        self.redraw_display()

    def redraw_display(self) -> None:
        if self.current_frame is None:
            return

        frame_h, frame_w = self.current_frame.shape[:2]
        canvas_h = frame_h + 2 * self.margin_px
        canvas_w = frame_w + 2 * self.margin_px
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[self.margin_px:self.margin_px + frame_h, self.margin_px:self.margin_px + frame_w] = self.current_frame

        cv2.putText(canvas, f"Frame {self.frame_index} | Zoom {self.zoom:.2f}x", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(canvas, "Left click: add point | a/d: prev/next | x: undo | +/-: zoom | 0: reset | Enter: solve (4 non-coplanar min) | q: quit", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        for i, ((u, v), (x, y, z)) in enumerate(zip(self.selected_points_2d, self.selected_points_3d)):
            px = int(round(u)) + self.margin_px
            py = int(round(v)) + self.margin_px
            pt = (px, py)
            cv2.circle(canvas, pt, 6, (0, 255, 255), -1)

            line1 = f"{i}: ({u:.1f}, {v:.1f})"
            line2 = f"-> [{x:.3f}, {y:.3f}, {z:.3f}]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 2
            (w1, h1), _ = cv2.getTextSize(line1, font, scale, thickness)
            (w2, h2), _ = cv2.getTextSize(line2, font, scale, thickness)
            text_w = max(w1, w2)
            text_h = h1 + h2 + 10

            text_x = px + 10
            if text_x + text_w > canvas_w - 10:
                text_x = px - 10 - text_w
            text_x = max(10, text_x)

            text_y1 = py - 8
            text_y2 = py + 12
            if text_y1 - h1 < 10:
                text_y1 = py + h1 + 10
                text_y2 = text_y1 + h2 + 8
            if text_y2 > canvas_h - 10:
                text_y2 = canvas_h - 10
                text_y1 = text_y2 - h2 - 8

            cv2.putText(canvas, line1, (text_x, text_y1), font, scale, (0, 255, 255), thickness)
            cv2.putText(canvas, line2, (text_x, text_y2), font, scale, (0, 255, 255), thickness)

        if self.zoom != 1.0:
            interp = cv2.INTER_CUBIC if self.zoom > 1.0 else cv2.INTER_AREA
            disp = cv2.resize(canvas, None, fx=self.zoom, fy=self.zoom, interpolation=interp)
        else:
            disp = canvas

        self.current_display = disp

    def prompt_for_3d_point(self, pixel_x: int, pixel_y: int) -> Optional[Tuple[float, float, float]]:
        point_text = simpledialog.askstring(
            "3D Point",
            f"Frame {self.frame_index}, pixel ({pixel_x}, {pixel_y})\nEnter corresponding 3D point as x,y,z:"
        )
        if point_text is None:
            return None
        try:
            x_str, y_str, z_str = point_text.replace(" ", "").split(",")
            return float(x_str), float(y_str), float(z_str)
        except Exception:
            messagebox.showerror("Invalid Input", "3D point must be entered as x,y,z")
            return None

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.current_frame is None:
            return

        frame_h, frame_w = self.current_frame.shape[:2]
        raw_x = x / self.zoom - self.margin_px
        raw_y = y / self.zoom - self.margin_px

        if raw_x < 0 or raw_x >= frame_w or raw_y < 0 or raw_y >= frame_h:
            return

        pixel_x = int(round(raw_x))
        pixel_y = int(round(raw_y))
        point_3d = self.prompt_for_3d_point(pixel_x, pixel_y)
        if point_3d is None:
            return
        self.selected_points_2d.append((float(pixel_x), float(pixel_y)))
        self.selected_points_3d.append(point_3d)
        self.redraw_display()

    def interactive_pnp_annotation(self) -> None:
        self.load_pnp_video()
        cv2.namedWindow("PnP Annotation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("PnP Annotation", 1400, 900)
        cv2.setMouseCallback("PnP Annotation", self.mouse_callback)

        while True:
            if self.current_display is not None:
                cv2.imshow("PnP Annotation", self.current_display)
            key = cv2.waitKeyEx(20)

            if key in (ord('q'), 27):
                raise RuntimeError("PnP annotation cancelled by user.")
            elif key in (ord('d'), 2555904):
                self.set_frame(self.frame_index + 1)
            elif key in (ord('a'), 2424832):
                self.set_frame(self.frame_index - 1)
            elif key in (ord('x'), 8):
                if self.selected_points_2d:
                    self.selected_points_2d.pop()
                    self.selected_points_3d.pop()
                    self.redraw_display()
            elif key in (ord('+'), ord('=')):
                self.zoom = min(self.max_zoom, self.zoom * self.zoom_step)
                self.redraw_display()
            elif key in (ord('-'), ord('_')):
                self.zoom = max(self.min_zoom, self.zoom / self.zoom_step)
                self.redraw_display()
            elif key == ord('0'):
                self.zoom = 1.0
                self.redraw_display()
            elif key in (13, 10):
                if len(self.selected_points_2d) < 4:
                    messagebox.showwarning("Not Enough Points", "At least 4 correspondences are required for PnP. For a generic unique solution, they should be non-coplanar.")
                    continue
                break

        cv2.destroyWindow("PnP Annotation")
        if self.cap_pnp is not None:
            self.cap_pnp.release()

    def solve_pnp(self) -> None:
        if self.K is None or self.dist_coeffs is None:
            raise RuntimeError("Intrinsics are not available.")

        object_points = np.array(self.selected_points_3d, dtype=np.float32).reshape(-1, 1, 3)
        image_points = np.array(self.selected_points_2d, dtype=np.float32).reshape(-1, 1, 2)

        if len(self.selected_points_2d) == 4:
            success, rvecs, tvecs = cv2.solvePnPGeneric(
                object_points,
                image_points,
                self.K,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_P3P,
            )[:3]
            if not success or len(rvecs) == 0:
                raise RuntimeError("solvePnPGeneric(P3P) failed.")
            self.rvec = rvecs[0]
            self.tvec = tvecs[0]
            return

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.K,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success:
            raise RuntimeError("solvePnP failed.")
        self.rvec = rvec
        self.tvec = tvec

    def update_json_file(self) -> Dict[str, Any]:
        if self.K is None or self.rvec is None or self.tvec is None:
            raise RuntimeError("Calibration result is incomplete.")
        cameras = load_camera_db(self.json_path)
        entry = find_camera_entry(cameras, self.camera_name)
        if entry is None:
            entry = {"name": self.camera_name}
            cameras.append(entry)
        entry["K"] = ndarray_to_K_list(self.K)
        entry["tvec"] = ndarray_to_tvec_list(self.tvec)
        entry["rvec"] = ndarray_to_rvec_row_list(self.rvec)
        save_camera_db(self.json_path, cameras)
        return entry

    def run(self) -> None:
        try:
            if not self.resolve_camera_name_and_intrinsics():
                return
            if not self.ask_inputs():
                return
            self.interactive_pnp_annotation()
            self.solve_pnp()
            entry = self.update_json_file()
            print_camera_entry(entry)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            print(f"Error: {exc}")
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            if self.cap_pnp is not None:
                self.cap_pnp.release()
            self.root.quit()
            self.root.destroy()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python extrinsic_calibration_gui.py <camera_json_path>")
    app = ExtrinsicCalibrationGUI(sys.argv[1])
    app.run()
