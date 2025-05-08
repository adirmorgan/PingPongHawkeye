import cv2
import numpy as np

def browse_npy_File(npy_path: str):
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

if __name__ == '__main__':
    browse_npy_File("C:\\Users\\elad2\\Downloads\\tryinnn_no_mask.npy")