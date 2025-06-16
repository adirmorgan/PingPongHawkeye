
import cv2
import numpy as np
from typing import Tuple
import cv2
import numpy as np
import json
from typing import Tuple

import cv2
import numpy as np
import json
from typing import Tuple

def Save_Video_To_npy_File_With_Or_without_HSV_Mask(
    video_path: str,
    hsv_lower: Tuple[int, int, int] = None,
    hsv_upper: Tuple[int, int, int] = None,
    max_frames: int = 300,
    output_path: str = "C:\\Users\\itays\\Downloads\\pingpong.npy",
    start_msec: int = 0,
    apply_mask: bool = False,
    config_path: str = None
):
    """
    Processes a video and optionally applies an HSV-based mask, then saves the frames as a NumPy array.
    If config_path is provided, parameters are loaded from the config file.

    Args:
        video_path (str): Path to the input video file.
        hsv_lower (Tuple[int, int, int]): Lower HSV bounds.
        hsv_upper (Tuple[int, int, int]): Upper HSV bounds.
        max_frames (int): Maximum frames to process.
        output_path (str): Path to save the .npy file.
        start_msec (int): Start position in ms.
        apply_mask (bool): Whether to apply HSV mask.
        config_path (str): Optional path to JSON config file.

    Raises:
        IOError: If the video cannot be opened.
    """

    if config_path:
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
    Save_Video_To_npy_File_With_Or_without_HSV_Mask(video_path="C:\\Users\\itays\\Downloads\\Game.mp4")