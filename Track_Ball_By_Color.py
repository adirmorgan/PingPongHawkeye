import cv2
import numpy as np
from typing import List, Tuple

def detect_ball(frame: np.ndarray,
                roi_bounds: Tuple[int, int, int, int],
                min_size: int,
                max_size: int,
                min_v_brightness: int,
                threshold_value: int) -> List[Tuple[int, int, int, int]]:
    """
    Detects ball candidates in a frame based on ROI, size, brightness, and thresholding.

    Args:
        frame (np.ndarray): The input frame in BGR format.
        roi_bounds (Tuple[int, int, int, int]): The region of interest (x_min, x_max, y_min, y_max).
        min_size (int): Minimum width/height of detected contour.
        max_size (int): Maximum width/height of detected contour.
        min_v_brightness (int): Minimum mean V-channel brightness to qualify as ball.
        threshold_value (int): Threshold value for binary segmentation (0-255).

    Returns:
        List[Tuple[int, int, int, int]]: A list of bounding boxes (x, y, w, h) for detected balls.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = []

    roi_x_min, roi_x_max, roi_y_min, roi_y_max = roi_bounds

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if min_size <= w <= max_size and min_size <= h <= max_size:
            if roi_x_min <= x <= roi_x_max and roi_y_min <= y <= roi_y_max:
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)

                v_channel = hsv[:, :, 2]
                mean_v = cv2.mean(v_channel, mask=mask)[0]

                if mean_v > min_v_brightness:
                    detected.append((x, y, w, h))

    return detected

def draw_ball_detections(frame: np.ndarray,
                         detections: List[Tuple[int, int, int, int]],
                         roi_bounds: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Draws rectangles around detected balls and the ROI boundary on the frame.

    Args:
        frame (np.ndarray): The input frame.
        detections (List[Tuple[int, int, int, int]]): List of bounding boxes for detected balls.
        roi_bounds (Tuple[int, int, int, int]): The region of interest bounds.

    Returns:
        np.ndarray: The output frame with drawings.
    """
    output = frame.copy()
    for (x, y, w, h) in detections:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle

    roi_x_min, roi_x_max, roi_y_min, roi_y_max = roi_bounds
    cv2.rectangle(output, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (255, 0, 255), 2)  # Magenta ROI box

    return output

def browse_frames_with_detection(
    npy_path: str,
    roi_bounds: Tuple[int, int, int, int],
    min_size: int = 7,
    max_size: int = 20,
    min_v_brightness: int = 210,
    threshold_value: int = 15
):
    """
    Loads frames from a .npy file and allows interactive browsing with ball detection overlays.

    Args:
        npy_path (str): Path to the .npy file containing the frames.
        roi_bounds (Tuple[int, int, int, int]): Region of interest bounds (x_min, x_max, y_min, y_max).
        min_size (int, optional): Minimum width/height for detected objects. Defaults to 7.
        max_size (int, optional): Maximum width/height for detected objects. Defaults to 20.
        min_v_brightness (int, optional): Minimum V-channel brightness to validate an object. Defaults to 210.
        threshold_value (int, optional): Grayscale threshold to generate binary image. Defaults to 15.

    Controls:
        'a' / Left Arrow - Previous frame
        'd' / Right Arrow - Next frame
        'q' - Quit
    """
    frames_array = np.load(npy_path)
    num_frames = frames_array.shape[0]
    index = 0

    window_name = "Ball Detection Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def show_frame(i):
        frame = frames_array[i]
        detections = detect_ball(frame, roi_bounds, min_size, max_size, min_v_brightness, threshold_value)
        display = draw_ball_detections(frame, detections, roi_bounds)
        cv2.imshow(window_name, display)

    def on_trackbar(val):
        nonlocal index
        index = val
        show_frame(index)

    cv2.createTrackbar("Frame", window_name, 0, num_frames - 1, on_trackbar)

    show_frame(index)

    print("Controls:\n  ← / a = Previous Frame\n  → / d = Next Frame\n  q = Quit")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('d'), 83, 0x27]:  # Right
            index = (index + 1) % num_frames
        elif key in [ord('a'), 81, 0x25]:  # Left
            index = (index - 1) % num_frames

        cv2.setTrackbarPos("Frame", window_name, index)
        show_frame(index)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    browse_frames_with_detection(
        npy_path="C:\\Users\\elad2\\Downloads\\filtered_frames.npy",
        roi_bounds=(320, 900, 200, 470),
        min_size=7,
        max_size=20,
        min_v_brightness=210,
        threshold_value=15
    )
