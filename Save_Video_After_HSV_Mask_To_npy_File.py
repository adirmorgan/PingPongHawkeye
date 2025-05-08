
import cv2
import numpy as np
from typing import Tuple
import cv2
import numpy as np
import json
from typing import Tuple

def Save_Video_To_npy_File_With_Or_without_HSV_Mask(
    video_path: str = None,
    hsv_lower: Tuple[int, int, int] = None,
    hsv_upper: Tuple[int, int, int] = None,
    max_frames: int = 300,
    output_path: str = "C:\\Users\\elad2\\Downloads\\filtered_frames.npy",
    start_msec: int = 20500,
    apply_mask: bool = True,
    use_config: bool = False,
    config_path: str = "config.json"
):
    """
    Processes a video and optionally applies an HSV-based mask, then saves the frames as a NumPy array.
    Supports optional loading of parameters from a JSON config file.

    Args:
        video_path (str): Path to the input video file.
        hsv_lower (Tuple[int, int, int]): Lower HSV bounds.
        hsv_upper (Tuple[int, int, int]): Upper HSV bounds.
        max_frames (int): Maximum frames to process.
        output_path (str): Path to save the .npy file.
        start_msec (int): Start position in ms.
        apply_mask (bool): Whether to apply HSV mask.
        use_config (bool): If True, parameters are loaded from config file.
        config_path (str): Path to JSON config file.

    Raises:
        IOError: If the video cannot be opened.
    """

    if use_config:
        with open(config_path, 'r') as f:
            config = json.load(f)
        video_path = config["video_path"]
        hsv_lower = tuple(config["hsv_lower"])
        hsv_upper = tuple(config["hsv_upper"])
        max_frames = config.get("max_frames", max_frames)
        output_path = config.get("output_path", output_path)
        start_msec = config.get("start_msec", start_msec)
        apply_mask = config.get("apply_mask", apply_mask)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)
    frames = []
    count = 0

    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if apply_mask:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
            filtered = cv2.bitwise_and(frame, frame, mask=mask)
            frames.append(filtered)
        else:
            frames.append(frame)

        count += 1

    cap.release()

    if frames:
        frames_array = np.array(frames, dtype=np.uint8)
        np.save(output_path, frames_array)
        print(f"Saved {len(frames)} {'masked' if apply_mask else 'original'} frames to {output_path}")
    else:
        print("No frames were processed and saved.")




if __name__ == '__main__':

    #example for config JSON:
    # {
    #     "video_path": "example_video.mp4",
    #     "hsv_lower": [30, 50, 50],
    #     "hsv_upper": [90, 255, 255],
    #     "max_frames": 200,
    #     "output_path": "output_frames.npy",
    #     "start_msec": 15000,
    #     "apply_mask": true
    # }

    Save_Video_To_npy_File_With_Or_without_HSV_Mask("C:\\Users\\elad2\\Downloads\\pingpong_720p60_final.mp4", (0, 0, 195), (179, 80, 255),output_path = "C:\\Users\\elad2\\Downloads\\tryinnn_no_mask.npy",apply_mask=False)