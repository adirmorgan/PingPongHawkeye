import cv2
import numpy as np
from typing import Tuple

def browse_npy_file(npy_path: str):
    """
    Loads a saved NumPy array of frames and allows interactive browsing using OpenCV.
    Args:
        npy_path (str): Path to the .npy file containing the video frames.
    Controls:
        'a' or Left Arrow (←): Previous frame
        'd' or Right Arrow (→): Next frame
        'q': Quit
    """

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



def Save_Video_After_HSV_Mask_To_npy_File(
    video_path: str,
    hsv_lower: Tuple[int, int, int],
    hsv_upper: Tuple[int, int, int],
    max_frames: int = 300,
    output_path: str = ".\\Data\\test_data\\test_vid.npy",
    start_msec: int = 20500
):
    """
    Processes a video to filter frames based on specified HSV color thresholds and saves them.

    This function loads a video starting from a specified timestamp, applies an HSV-based color
    filter to each frame, and saves the filtered frames as a NumPy array (.npy file) for further processing.

    Args:
        video_path (str): Path to the input video file.
        hsv_lower (Tuple[int, int, int]): Lower HSV bounds (Hue, Saturation, Value).
        hsv_upper (Tuple[int, int, int]): Upper HSV bounds (Hue, Saturation, Value).
        max_frames (int, optional): Maximum number of frames to process. Defaults to 300.
        output_path (str, optional): Path to save the filtered frames (.npy file). Defaults to 'C:\\Users\\elad2\\Downloads\\filtered_frames.npy'.
        start_msec (int, optional): Start time in milliseconds to begin reading the video. Defaults to 20500 ms.

    Raises:
        IOError: If the video cannot be opened.

    Outputs:
        Saves a NumPy array containing the filtered frames at the specified output path.
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

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
        filtered = cv2.bitwise_and(frame, frame, mask=mask)
        frames.append(filtered)
        count += 1

    cap.release()

    if frames:
        frames_array = np.array(frames, dtype=np.uint8)
        np.save(output_path, frames_array)
        print(f"Saved {len(frames)} filtered frames to {output_path}")
    else:
        print("No frames were processed and saved.")


if __name__ == '__main__':
    Save_Video_After_HSV_Mask_To_npy_File(".\\Data\\sim_data\\camera1.mp4",
                                          (0, 0, 195), (179, 80, 255),
                                          output_path = ".\\Data\\test_data\\test_processed_vid.npy")

    browse_npy_file(".\\Data\\test_data\\test_processed_vid.npy")

