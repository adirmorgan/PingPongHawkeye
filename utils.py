import cv2
import numpy as np
from typing import Tuple
import os, sys

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


if __name__ == '__main__':
    # vid_array = load_video_to_array("Data/sim_data/camera1.mp4")
    # print("loaded video to np array")
    #
    # output_path = "Data/test_data/test_processed_vid.npy"
    # save_video_as_array(vid_array, output_path)
    # print("saved np array")


    browse_npy_file("Data/test_data/test_processed_vid.npy")

