import cv2
import numpy as np
from typing import List, Tuple



def browse_contours_from_npy(npy_path: str,
                              roi_bounds: Tuple[int, int, int, int],
                              min_size: int,
                              max_size: int,
                              min_v_brightness: int,
                              max_s_saturation: int,
                              threshold_value: int,
                              window_name: str = "Contours Viewer") -> None:
    """
    Visualizes contours from a 4D NumPy array (video frames) with filtering by ROI, size, V-channel brightness,
    and S-channel saturation. The user can interactively browse through frames and see detected contours.

    Args:
        npy_path (str): Path to a .npy file containing a 4D array of video frames with shape
                        (frames, height, width, channels), in BGR format.
        roi_bounds (Tuple[int, int, int, int]): Region of interest bounds as (x_min, x_max, y_min, y_max).
                                                Only contours within these bounds are considered.
        min_size (int): Minimum width and height (in pixels) of a valid contour's bounding box.
        max_size (int): Maximum width and height (in pixels) of a valid contour's bounding box.
        min_v_brightness (int): Minimum mean value (V-channel) within the bounding box for a contour to be valid.
        max_s_saturation (int): Maximum allowed mean saturation (S-channel) within the bounding box.
                                This helps reject saturated (colored) regions.
        threshold_value (int): Threshold value (0–255) for binary segmentation on the grayscale image.
        window_name (str): Name of the OpenCV window. Default is "Contours Viewer".

    Displays:
        - Bounding box (in blue) around each valid contour.
        - Filled contour (in green).
        - Mean V value (in green) as a label above the contour.
        - Mean S value (in red) as a label below the contour.

    Controls:
        - Left / 'a': Previous frame
        - Right / 'd': Next frame
        - 'q': Quit viewer
        - Trackbar: Jump to specific frame
    """

    # Load frames from npy file
    try:
        frames_array = np.load(npy_path)
    except Exception as e:
        raise IOError(f"Failed to load frames from {npy_path}: {e}")

    # Ensure the array is 4D (frames, height, width, channels)
    if frames_array.ndim != 4:
        raise ValueError("Expected a 4D array (frames, height, width, channels)")

    num_frames = frames_array.shape[0]
    index = 0
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def show_frame(i: int):
        # Get a copy of the current frame
        frame = frames_array[i].copy()

        # Convert frame to grayscale and apply binary thresholding
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Find external contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert frame to HSV to extract V and S channels
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]

        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)

            # Size filter
            if min_size <= w <= max_size and min_size <= h <= max_size:
                # ROI filter
                if roi_bounds[0] <= x <= roi_bounds[1] and roi_bounds[2] <= y <= roi_bounds[3]:
                    # Extract V and S from bounding box
                    roi_v = v_channel[y:y+h, x:x+w]
                    roi_s = s_channel[y:y+h, x:x+w]
                    mean_v = np.mean(roi_v)
                    mean_s = np.mean(roi_s)

                    # Filter by brightness and saturation
                    if mean_v > min_v_brightness and mean_s < max_s_saturation:
                        # Draw bounding rectangle (blue)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

                        # Fill contour (green)
                        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), -1)

                        # Text for V channel (above contour) – green
                        text_v = f"{idx} | V={mean_v:.1f}"
                        cv2.putText(frame, text_v, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

                        # Text for S channel (below contour) – red
                        text_s = f"S={mean_s:.1f}"
                        cv2.putText(frame, text_s, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Show the processed frame
        cv2.imshow(window_name, frame)

    # Callback for trackbar navigation
    def on_trackbar(val: int):
        nonlocal index
        index = val
        show_frame(index)

    # Create trackbar to browse through frames
    cv2.createTrackbar("Frame", window_name, 0, num_frames - 1, on_trackbar)
    show_frame(index)

    print("Controls:\n  ← / a = Previous\n  → / d = Next\n  q = Quit")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('d'), 83, 0x27]:  # Right / d key
            index = (index + 1) % num_frames
        elif key in [ord('a'), 81, 0x25]:  # Left / a key
            index = (index - 1) % num_frames

        cv2.setTrackbarPos("Frame", window_name, index)
        show_frame(index)

    # Cleanup
    cv2.destroyAllWindows()



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

def detect_ball_ver2(frame: np.ndarray,
                     roi_bounds: Tuple[int, int, int, int],
                     min_size: int,
                     max_size: int,
                     min_v_brightness: int,
                     max_s_saturation: int,
                     threshold_value: int) -> List[Tuple[int, int, int, int]]:
    """
    Detects ball candidates in a frame based on ROI, size, brightness, saturation, and thresholding.
    Uses bounding box for V and S channel mean calculation.

    Args:
        frame (np.ndarray): The input frame in BGR format.
        roi_bounds (Tuple[int, int, int, int]): The region of interest (x_min, x_max, y_min, y_max).
        min_size (int): Minimum width/height of detected contour.
        max_size (int): Maximum width/height of detected contour.
        min_v_brightness (int): Minimum mean V-channel brightness to qualify as ball.
        min_s_saturation (int): Minimum mean S-channel saturation to qualify as ball.
        threshold_value (int): Threshold value for binary segmentation (0-255).

    Returns:
        List[Tuple[int, int, int, int]]: A list of bounding boxes (x, y, w, h) for detected balls.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = []

    roi_x_min, roi_x_max, roi_y_min, roi_y_max = roi_bounds

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if min_size <= w <= max_size and min_size <= h <= max_size:
            if roi_x_min <= x <= roi_x_max and roi_y_min <= y <= roi_y_max:
                roi_v = v_channel[y:y+h, x:x+w]
                roi_s = s_channel[y:y+h, x:x+w]

                mean_v = np.mean(roi_v)
                mean_s = np.mean(roi_s)

                if mean_v > min_v_brightness and mean_s < max_s_saturation:
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
    max_s_saturation: int=28,
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
        detections = detect_ball_ver2(frame, roi_bounds, min_size, max_size, min_v_brightness,max_s_saturation, threshold_value)
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
        npy_path="C:\\Users\\elad2\\Downloads\\tryinnn.npy",
        roi_bounds=(320, 900, 200, 470),
        min_size=7,
        max_size=20,
        min_v_brightness=100,
        max_s_saturation=30,
        threshold_value=30
    )
