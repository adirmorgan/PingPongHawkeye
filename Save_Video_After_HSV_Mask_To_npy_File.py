
import cv2
import numpy as np
from typing import Tuple

def Save_Video_To_npy_File_With_Or_without_HSV_Mask(
    video_path: str,
    hsv_lower: Tuple[int, int, int],
    hsv_upper: Tuple[int, int, int],
    max_frames: int = 300,
    output_path: str = "C:\\Users\\elad2\\Downloads\\filtered_frames.npy",
    start_msec: int = 20500,
    apply_mask: bool = True
):
    """
    Processes a video and optionally applies an HSV-based mask, then saves the frames as a NumPy array.

    Args:
        video_path (str): Path to the input video file.
        hsv_lower (Tuple[int, int, int]): Lower HSV bounds (Hue, Saturation, Value).
        hsv_upper (Tuple[int, int, int]): Upper HSV bounds (Hue, Saturation, Value).
        max_frames (int, optional): Maximum number of frames to process. Defaults to 300.
        output_path (str, optional): Path to save the frames (.npy file). Defaults to 'C:\\Users\\elad2\\Downloads\\filtered_frames.npy'.
        start_msec (int, optional): Start time in milliseconds to begin reading the video. Defaults to 20500 ms.
        apply_mask (bool, optional): Whether to apply the HSV mask before saving. Defaults to True.

    Raises:
        IOError: If the video cannot be opened.
    """
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
    Save_Video_To_npy_File_With_Or_without_HSV_Mask("C:\\Users\\elad2\\Downloads\\pingpong_720p60_final.mp4", (0, 0, 195), (179, 80, 255),output_path = "C:\\Users\\elad2\\Downloads\\tryinnn_no_mask.npy",apply_mask=False)