from __future__ import annotations

import json
import concurrent.futures
from typing import Any, Callable, Generic, Iterable, Iterator, Optional, TypeVar
from cv2 import Mat
from numpy import ndarray, dtype, float64

from typing import List, Tuple, Any
import numpy as np

import time
import threading
from functools import wraps

# controled by the top module (it's main should call the function "timing")
TIMING_ENABLED = False # unless requested explicitly (using the timing() method) - no timing prints will be shown

def timing(enable: bool = True) -> None:
    """Global switch for timeit (set from config)."""
    global TIMING_ENABLED
    TIMING_ENABLED = bool(enable)


class timeit:

    # thread-local depth so each thread gets its own tree
    _state = threading.local()

    @classmethod
    def _get_depth(cls) -> int:
        return getattr(cls._state, "depth", 0)

    @classmethod
    def _set_depth(cls, value: int) -> None:
        cls._state.depth = max(0, value)

    @classmethod
    def _push(cls) -> None:
        cls._set_depth(cls._get_depth() + 1)

    @classmethod
    def _pop(cls) -> None:
        cls._set_depth(cls._get_depth() - 1)

    def __init__(self, label=None):
        self.label = label
        self._start = None
        self._enabled_snapshot = None
        self._used = False

    # ==== context manager: with timeit("...") ====
    def __enter__(self):

        self._enabled_snapshot = bool(TIMING_ENABLED)
        if self._enabled_snapshot:
            self._used = True
            type(self)._push()
            self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._used:
            depth_for_print = type(self)._get_depth() - 1  # העומק של הבלוק הזה
            if self._enabled_snapshot and self._start is not None:
                elapsed = time.perf_counter() - self._start
                indent = "\t" * max(depth_for_print, 0)
                label = self.label or "Block"
                printGreen(f"{indent}{label} took {elapsed:.6f} seconds")
            type(self)._pop()
        return False

    # ==== decorator: @timeit("...") or @timeit ====
    def __call__(self, func):
        outer_label = self.label

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not TIMING_ENABLED:
                return func(*args, **kwargs)
            label = outer_label or func.__name__
            with timeit(label):
                return func(*args, **kwargs)

        return wrapper



T = TypeVar("T")
R = TypeVar("R")

# Global toggle for threadit
THREADING_ENABLE: bool = True


def threading(enable: bool) -> None:
    """
    Enable/disable multithreading globally for threadit.

    Usage:
        threading(True)   # enable
        threading(False)  # disable (threadit behaves sequentially)
    """
    global THREADING_ENABLE
    THREADING_ENABLE = bool(enable)


class _ImmediateFuture(Generic[R]):
    """
    Future-like object used when threading is disabled.
    Supports .result() so the same code works in both modes.
    """
    __slots__ = ("_value", "_exc")

    def __init__(self, value: Optional[R] = None, exc: Optional[BaseException] = None):
        self._value = value
        self._exc = exc

    def result(self, timeout: Optional[float] = None) -> R:
        if self._exc is not None:
            raise self._exc
        return self._value  # type: ignore[return-value]


class threadit:
    """
    Context manager that provides a thread-pool interface while allowing a global on/off switch.

    When THREADING_ENABLE is False:
      - submit() executes immediately and returns an _ImmediateFuture
      - map() runs sequentially
      - as_completed() yields futures in input order

    When THREADING_ENABLE is True:
      - uses ThreadPoolExecutor (either provided or created locally)

    Patterns:

        with threadit(max_workers=16) as th:
            futs = [th.submit(work, x) for x in items]
            for fut in th.as_completed(futs):
                out = fut.result()

        with threadit(max_workers=16) as th:
            for out in th.map(work, items):
                ...

    IMPORTANT:
      - threadit does NOT print timing by itself; wrap it with your existing timeit if you want timing.
    """

    def __init__(
        self,
        *,
        max_workers: Optional[int] = None,
        executor: Optional[concurrent.futures.Executor] = None,
        enabled: Optional[bool] = None,
    ):
        self.max_workers = max_workers
        self._external_executor = executor
        self._executor: Optional[concurrent.futures.Executor] = None
        self._owns_executor = False
        self._enabled = THREADING_ENABLE if enabled is None else bool(enabled)

    def __enter__(self) -> "threadit":
        if self._enabled:
            if self._external_executor is not None:
                self._executor = self._external_executor
                self._owns_executor = False
            else:
                # Reasonable default if not specified
                if self.max_workers is None:
                    cpu = os.cpu_count() or 1
                    mw = min(32, cpu + 4)
                else:
                    mw = int(self.max_workers)
                    if mw <= 0:
                        mw = 1

                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=mw)
                self._owns_executor = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._owns_executor and self._executor is not None:
            self._executor.shutdown(wait=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def submit(self, fn: Callable[..., R], /, *args: Any, **kwargs: Any):
        """
        Submit a task; returns a Future-like object:
          - concurrent.futures.Future when enabled
          - _ImmediateFuture when disabled
        """
        if not self._enabled:
            try:
                return _ImmediateFuture(fn(*args, **kwargs))
            except BaseException as e:
                return _ImmediateFuture(exc=e)

        if self._executor is None:
            raise RuntimeError("threadit.submit() used outside of 'with threadit(...)' context.")
        return self._executor.submit(fn, *args, **kwargs)

    def map(self, fn: Callable[[T], R], iterable: Iterable[T]) -> Iterator[R]:
        """
        Like executor.map, but sequential when disabled.
        """
        if not self._enabled:
            for x in iterable:
                yield fn(x)
            return

        if self._executor is None:
            raise RuntimeError("threadit.map() used outside of 'with threadit(...)' context.")
        yield from self._executor.map(fn, iterable)

    def as_completed(self, futures: Iterable[Any]) -> Iterator[Any]:
        """
        Yield futures as they complete (enabled) or in input order (disabled).
        """
        if not self._enabled:
            for f in futures:
                yield f
            return
        yield from concurrent.futures.as_completed(futures)
def printGreen(s): print("\033[92m {}\033[00m".format(s))
def printGrey(s): print("\033[37m {}\033[00m".format(s))
def Contours(frames: np.ndarray, frame_index: int, cfg: dict) -> List[Tuple[np.ndarray, int]]:
    """
    Return all contours in the given frame after masking via your shape thresholds.

    Args:
      frames (np.ndarray): video frames array.
      frame_index (int): index of the current frame.
      cfg (dict): shape_detection config section (color_space, thresholds, lab settings, etc.)

    Returns:
      List of contours
    """
    frame = frames[frame_index]
    match cfg["contours"]["mask_method"]:
        case "motion":
            mask = motion_mask(frames, frame_index, cfg.get('motion_k', 5), 0,
                                  shadow_weight=cfg.get('shadow_weight', 0.3))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours
        case "shape":
            mask = shape_mask(frame, cfg)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours
        case "color":
            mask = color_mask(frame, cfg)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours
        case _:
            raise ValueError(f"Invalid mask method: {cfg['contours']['mask_method']}")

def get_coordinates(contour):
    """
    Compute centroid (x, y) of given contour using image moments.
    """
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        x = int(moments['m10'] / moments['m00'])
        y = int(moments['m01'] / moments['m00'])
        return (x, y)
    return None


def create_config_json_file_for_Save_npy_file(
        video_path: str,
        hsv_lower: tuple,
        hsv_upper: tuple,
        output_path: str,
        max_frames: int = 300,
        start_msec: int = 20500,
        apply_mask: bool = True,
        file_path: str = "config2.json"
):
    """
    Creates a JSON config file with the provided video processing parameters.

    Args:
        video_path (str): Path to the input video.
        hsv_lower (tuple): Lower HSV bounds (H, S, V).
        hsv_upper (tuple): Upper HSV bounds (H, S, V).
        output_path (str): Path to save the output .npy file.
        max_frames (int): Max frames to process. Defaults to 300.
        start_msec (int): Start time in ms. Defaults to 20500.
        apply_mask (bool): Whether to apply HSV mask. Defaults to True.
        file_path (str): Path to write the config JSON file. Defaults to 'config.json'.
    """
    config = {
        "video_path": video_path,
        "hsv_lower": list(hsv_lower),
        "hsv_upper": list(hsv_upper),
        "max_frames": max_frames,
        "output_path": output_path,
        "start_msec": start_msec,
        "apply_mask": apply_mask
    }

    with open(file_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Config file saved to {file_path}")


def browse_npy_file(npy_path: str, window_name:str = 'Video'):
    try:
        frames_array = np.load(npy_path)
    except Exception as e:
        raise IOError(f"Failed to load frames from {npy_path}: {e}")

    if frames_array.ndim != 4:
        raise ValueError(f"Expected a 4D array (frames, height, width, channels), got {frames_array.ndim}D array.")

    num_frames = frames_array.shape[0]
    if num_frames == 0:
        raise ValueError("No frames found in the loaded array.")

    index = 0
    window_name = "Filtered Video"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def show_frame(i: int):
        frame = frames_array[i]
        cv2.imshow(window_name, frame)

    def on_trackbar(val: int):
        nonlocal index
        index = val
        show_frame(index)

    cv2.createTrackbar("Frame", window_name, 0, num_frames - 1, on_trackbar)
    show_frame(index)
    print("Controls:\n  ← / a = Previous\n  → / d = Next\n  q = Quit")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('d'), 83, 0x27]:  # 'd' or Right Arrow
            index = (index + 1) % num_frames
        elif key in [ord('a'), 81, 0x25]:  # 'a' or Left Arrow
            index = (index - 1) % num_frames

        cv2.setTrackbarPos("Frame", window_name, index)
        show_frame(index)

    cv2.destroyAllWindows()

def browse_contours_with_detection(frames_array: np.ndarray,
                                   roi_bounds: Tuple[int, int, int, int],
                                   min_size: int,
                                   max_size: int,
                                   min_v_brightness: int,
                                   threshold_value: int,
                                   window_name: str = "Contours Viewer") -> None:
    """
    Browse frames with contours drawn in green, and fill detected contours based on detection logic.

    Args:
        frames_array (np.ndarray): 4D numpy array (frames, height, width, channels).
        roi_bounds, min_size, max_size, min_v_brightness, threshold_value: Parameters for detect_ball.
        window_name (str): Name of the display window.
    """
    if frames_array.ndim != 4:
        raise ValueError("Expected a 4D array (frames, height, width, channels)")

    num_frames = frames_array.shape[0]
    index = 0
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def show_frame(i: int):
        frame = frames_array[i].copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if min_size <= w <= max_size and min_size <= h <= max_size:
                if roi_bounds[0] <= x <= roi_bounds[1] and roi_bounds[2] <= y <= roi_bounds[3]:
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    mean_v = cv2.mean(v_channel, mask=mask)[0]
                    if mean_v > min_v_brightness:
                        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), -1)  # Fill contour green

        cv2.imshow(window_name, frame)

    def on_trackbar(val: int):
        nonlocal index
        index = val
        show_frame(index)

    cv2.createTrackbar("Frame", window_name, 0, num_frames - 1, on_trackbar)
    show_frame(index)
    print("Controls:\n  ← / a = Previous\n  → / d = Next\n  q = Quit")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('d'), 83, 0x27]:
            index = (index + 1) % num_frames
        elif key in [ord('a'), 81, 0x25]:
            index = (index - 1) % num_frames
        cv2.setTrackbarPos("Frame", window_name, index)
        show_frame(index)

    cv2.destroyAllWindows()



def load_video_to_array(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    # Convert list of frames to a 4D numpy array (num_frames, height, width, channels)
    return np.array(frames)

def save_video_as_array(vid_array, output_path):
    if vid_array is not None:
        frames_array = np.array(vid_array, dtype=np.uint8)
        np.save(output_path, frames_array)
        print(f"Saved {len(vid_array)} filtered frames to {output_path}")
    else:
        print("No frames were processed and saved.")


import cv2
import numpy as np
import os

def stream_and_save_video_as_array(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_list = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_list.append(frame.astype(np.uint8))
        frame_count += 1

    cap.release()

    frames_array = np.array(frame_list, dtype=np.uint8)
    np.save(output_path, frames_array)
    print(f"Saved {frame_count} frames to {output_path}")


def toleround(x, tolerance):
    """
    Round a number or a list of numbers to the nearest multiple of `tolerance`.
    """
    def _round_one(num: float) -> float:
        return round(num / tolerance) * tolerance

    # If it's a list (or tuple), round element-wise
    if isinstance(x, (list, tuple)):
        return [_round_one(n) for n in x]
    else:
        # Scalar case
        return _round_one(x)

def motion_mask(frames: np.ndarray,
                         frame_index: int,
                         k: int,
                         threshold: float,
                         shadow_weight: float) -> tuple[ndarray[Any, dtype[float64]] | ndarray[Any, Any] | ndarray[
    Any, dtype[Any]] | ndarray[tuple[int, ...], dtype[float64]] | ndarray[tuple[int, ...], dtype[Any]], ndarray[
                                                            Any, dtype[float64]] | ndarray[Any, Any] | ndarray[
                                                            Any, dtype[Any]] | ndarray[
                                                            tuple[int, ...], dtype[float64]] | ndarray[
                                                            tuple[int, ...], dtype[Any]]] | Mat | ndarray:
    """
    Compute:
      - binary motion mask via k-frame difference
      - continuous motion_strength in [0,1], where shadows are down-weighted.

    shadow_weight < 1.0 reduces the contribution of darkening (shadows).
    """
    h, w = frames[0].shape[:2]

    if frame_index < k:
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.float32)

    # Current and temporal average of previous k frames in grayscale
    curr = cv2.cvtColor(frames[frame_index], cv2.COLOR_BGR2GRAY).astype(np.float32)
    prev_stack = [
        cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        for i in range(frame_index - k, frame_index)
    ]
    base = sum(prev_stack) / float(k)

    # Directional difference
    diff = curr - base
    pos = np.clip(diff, 0, None)        # brightening (object)
    neg = np.clip(-diff, 0, None)       # darkening (shadow)

    # Shadows contribute less
    motion = (1-shadow_weight)*pos + shadow_weight * neg  # still >= 0

    # Binary mask for visualization / contour extraction
    mask = (motion > threshold).astype(np.uint8) * 255

    # Smooth and dilate a bit
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def color_mask(frame: np.ndarray, cfg: dict):
    # 1) convert to HSV and threshold for white
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg["color_detection"]['lower_hsv'], dtype=np.uint8)
    upper = np.array(cfg["color_detection"]['upper_hsv'], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)

    # 2) clean up small holes and noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask

def shape_mask(frame: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Generate binary mask for candidate ball regions based on color/Lab thresholds.
    Morphology is tuned to avoid merging nearby objects too aggressively.
    cfg keys (examples):
      - "color_space": "HSV" or "YCrCb"
      - "lower_thresh": [H, S, V] or [Y, Cr, Cb]
      - "upper_thresh": [H, S, V] or [Y, Cr, Cb]
      - "lab_thresh": float (optional)
      - "lab_ref": [L, a, b] (optional)
      - "gaussian_ksize": int (odd)
      - "open_ksize": int
      - "close_ksize": int
    """
    # Ensure 3 channels (convert grayscale to BGR if needed)
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Optional Lab-based distance thresholding
    lab_thresh = cfg.get("lab_thresh", None)
    if lab_thresh is not None:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab).astype(np.float32)
        ref = np.array(cfg.get("lab_ref", [200.0, 128.0, 128.0]), dtype=np.float32)
        dist = np.linalg.norm(lab - ref[None, None, :], axis=2)
        mask = (dist < float(lab_thresh)).astype(np.uint8) * 255
    else:
        # Default: HSV or YCrCb in-range threshold
        space = cfg.get("color_space", "HSV").upper()
        if space == "HSV":
            conv = cv2.COLOR_BGR2HSV
        elif space in ("YCRCB", "YCrCb"):
            conv = cv2.COLOR_BGR2YCrCb
        else:
            conv = cv2.COLOR_BGR2HSV  # fallback

        cs = cv2.cvtColor(frame, conv)
        lower = np.array(cfg.get("lower_thresh", [0, 0, 200]), dtype=np.uint8)
        upper = np.array(cfg.get("upper_thresh", [180, 30, 255]), dtype=np.uint8)
        mask = cv2.inRange(cs, lower, upper)

    # Light blur to smooth noise, but not too large to avoid merging objects
    gaussian_ksize = int(cfg.get("gaussian_ksize", 5))
    if gaussian_ksize % 2 == 0:
        gaussian_ksize += 1
    if gaussian_ksize >= 3:
        mask = cv2.GaussianBlur(mask, (gaussian_ksize, gaussian_ksize), 0)

    # Morphology: first OPEN (remove noise), then a small CLOSE (fix small holes)
    open_ksize = int(cfg.get("open_ksize", 3))
    close_ksize = int(cfg.get("close_ksize", 3))

    if open_ksize > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    if close_ksize > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask

class video_flow_controller:
    def __init__ (self, frame_idx = 0, speed_cnt=1, pause_flag=False, back_flag=False, delay=1, nframes=None):
        if type(frame_idx) != int:
            raise TypeError("frame_idx must be an integer")
        if type(speed_cnt) != int:
            raise TypeError("speed_cnt must be an integer")
        if type(delay) != int:
            raise TypeError("delay must be an integer")
        if delay <=0:
            raise ValueError("delay must be a positive integer")
        if nframes is not None and type(nframes) != int:
            raise TypeError("nframes must be an integer or None")
        if nframes is not None and nframes <= 0:
            raise ValueError("nframes must be a positive integer or None")

        self.frame_idx = frame_idx
        self.speed_cnt = speed_cnt
        self.pause_flag = pause_flag
        self.back_flag = back_flag
        self.delay = delay
        self.nframes = nframes

    # Setters
    def set_frame_index(self, frame_idx):
        self.frame_idx = frame_idx
    def set_speed(self, speed_cnt):
        self.speed_cnt = speed_cnt
    def set_pause(self, pause_flag):
        self.pause_flag = pause_flag
    def set_back(self, back_flag):
        self.back_flag = back_flag

    #Getters
    def get_frame_index(self):
        return self.frame_idx
    def get_speed(self):
        return self.speed_cnt
    def get_pause(self):
        return self.pause_flag
    def get_back(self):
        return self.back_flag

    # More methods
    def next_frame(self, frame_idx = None):
        if frame_idx is not None:
            self.frame_idx = frame_idx

        key = cv2.waitKey(self.delay) & 0xFF
        if key == ord("q"):
            print("Forced stop by the user")
            self.frame_idx = self.nframes # skip to the end
            return self.frame_idx
        if key in [ord("p"), ord(" "), ord("0")]:
            self.pause_flag = not self.pause_flag
        if key == ord("b") or key == ord("-"):
            self.back_flag = not self.back_flag
        if key == ord("f") or key == ord("+"):
            self.speed_cnt += 1
        if key == ord("r") or key == ord("="):
            self.speed_cnt = 1
            self.pause_flag = False
            self.back_flag = False
        if key in range(ord("1"), ord("9")):  # enabling up to x8 speedup
            self.speed_cnt = int(chr(key))

        # increment frame index with respect to current state
        increment = self.speed_cnt
        if self.pause_flag: increment *= 0
        if self.back_flag: increment *= -1
        self.frame_idx += increment
        return self.frame_idx

    def loop_cond(self, frame_idx = None):
        if frame_idx is not None: self.frame_idx = frame_idx
        if self.nframes is None: raise ValueError("Checking loop condition without knowing the total number of frames")
        if self.frame_idx < 0:
            return False
        if self.frame_idx >= self.nframes:
            return False
        return True

    def info_text(self):
        info_text = f"SPEED : x{self.speed_cnt}" if self.speed_cnt > 1 else ""
        info_text += f" | BACK" if self.back_flag else ""
        info_text += f" | PAUSE" if self.pause_flag else ""
        return info_text
