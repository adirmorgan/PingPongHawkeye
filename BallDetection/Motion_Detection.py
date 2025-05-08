import cv2
import numpy as np


import cv2
import numpy as np

def steps_differences_motion_detection(
        video_path: str,
        k: int = 5,
        threshold: int = 25
):
    """
    Run k-step difference motion detection on a single video using RGB vector norm.

    Args:
      video_path (str): Path to the input video file.
      k (int, optional): Number of frames back to compare. Defaults to 5.
      threshold (int, optional): Binary threshold for motion mask. Defaults to 25.
    """
    cap = cv2.VideoCapture(video_path)
    frame_buffer = []
    window_name = f"{k}-Step Motion Detection (RGB Norm)"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame.copy())
        if len(frame_buffer) > k:
            frame_buffer.pop(0)

        if len(frame_buffer) < k:
            motion_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        else:
            diff = frame.astype(np.float32) - frame_buffer[0].astype(np.float32)
            norm = np.linalg.norm(diff, axis=2)
            motion_mask = np.where(norm > threshold, 255, 0).astype(np.uint8)

        cv2.imshow(window_name, motion_mask)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def background_model_motion_detection(
        video_path: str,
        learning_rate: float = 0.01,
        threshold: int = 30
):
    """
    Run background-model motion detection on a single video.

    Args:
      video_path (str): Path to the input video file.
      learning_rate (float, optional): Alpha for cv2.accumulateWeighted. Defaults to 0.01.
      threshold (int, optional): Binary threshold for motion mask. Defaults to 30.
    """
    cap = cv2.VideoCapture(video_path)
    background = None
    window_name = "Background Model Motion Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Initialize or update 3-channel background
        if background is None:
            background = frame.astype("float")
        else:
            cv2.accumulateWeighted(frame, background, learning_rate)

        # Convert running background to uint8
        bg_frame = cv2.convertScaleAbs(background)

        # Compute per-channel absolute difference
        diff = cv2.absdiff(frame, bg_frame)

        # Threshold each channel and combine
        b, g, r = cv2.split(diff)
        _, mb = cv2.threshold(b, threshold, 255, cv2.THRESH_BINARY)
        _, mg = cv2.threshold(g, threshold, 255, cv2.THRESH_BINARY)
        _, mr = cv2.threshold(r, threshold, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.bitwise_or(cv2.bitwise_or(mb, mg), mr)

        cv2.imshow(window_name, motion_mask)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


def main():
    method = input(
        "Choose your motion detection method:\n [1] steps differences\n [2] background image \n Your choice: ")
    if method == "1":
        steps_differences_motion_detection(
            video_path=r'C:\Users\itays\PythonProjects\PingPongHawkeye\Data\sim_data\Shapes1.mp4',
            k=5,
            threshold=25
        )
    if method == "2":
        background_model_motion_detection(
            video_path=r'C:\Users\itays\PythonProjects\PingPongHawkeye\Data\sim_data\Shapes1.mp4',
            learning_rate=0.01,
            threshold=30
        )


if __name__ == '__main__':
    main()
