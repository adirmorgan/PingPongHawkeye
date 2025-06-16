import cv2
import numpy as np
import json
import cv2
from BallDetection.ShapeDetection import preprocess_mask
from typing import List, Tuple
import numpy as np

def Contours(frames: np.ndarray, frame_index: int, cfg: dict) -> List[Tuple[np.ndarray, int]]:
    """
    Return all contours in the given frame after masking via your shape thresholds.

    Args:
      frames (np.ndarray): video frames array.
      frame_index (int): index of the current frame.
      cfg (dict): shape_detection config section (color_space, thresholds, lab settings, etc.)

    Returns:
      List of (contour, idx) tuples.
    """
    frame = frames[frame_index]
    mask = preprocess_mask(frame, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [(cnt, i) for i, cnt in enumerate(contours)]

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



if __name__ == '__main__':
    create_config_json_file_for_Save_npy_file(
        video_path="C:\\Users\\elad2\\Downloads\\pingpong_720p60_final.mp4",
        hsv_lower=(0, 0, 195),
        hsv_upper=(179, 80, 255),
        output_path="C:\\Users\\elad2\\Downloads\\tryinnn_no_mask.npy",
        apply_mask=False,
        file_path="C:\\Users\\elad2\\Downloads\\config2.json"
    )




