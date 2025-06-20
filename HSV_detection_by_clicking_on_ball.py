
import numpy as np
import cv2
import csv
import os

def collect_hsv_of_moving_ball(npy_path: str, max_object_size: int = 200, output_csv_path: str = "hsv_clicks_summary.csv"):
    """
    Allows manual labeling of a moving object (typically a ball) in a video loaded from an .npy file.
    The user can click on moving objects in each frame. Only objects smaller than a given size are considered.
    Once an object is clicked, its HSV values are stored, visual feedback is given (green overlay),
    and its average HSV is saved to a CSV file.

    Args:
        npy_path (str): Path to the .npy file containing video frames as a numpy array.
        max_object_size (int): Maximum pixel count for a selected object to be accepted. Default is 200.
        output_csv_path (str): Path to a CSV file where average HSV values for each selected object will be stored.

    Returns:
        np.ndarray: An array of shape [N, 3] containing all collected HSV values from all selected objects.
    """

    # Load video frames
    frames = np.load(npy_path)

    # Initialize background subtractor to detect motion
    back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)

    # List to hold all collected HSV values (per pixel)
    selected_hsv_values = []

    print("Instructions:")
    print(f"- Click on the ball to select it.")
    print(f"- Only objects with <= {max_object_size} pixels will be saved.")
    print(f"- The average HSV of each selection will be written to {output_csv_path}.")
    print("- Press Enter to go to the next frame.")
    print("- Press 'q' or 'Esc' to exit at any time.")

    # Keeps track of last selected object in the frame
    current_click = {"mask": None, "contour": None}

    # Initialize CSV file and write header
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "mean_H", "mean_S", "mean_V"])

    def click_event(event, x, y, flags, param):
        """
        Mouse click callback to handle object selection.
        If the user clicks on a moving object, extract HSV values and store them.
        """
        nonlocal current_click, frame_index
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv_frame, contours = param
            for cnt in contours:
                if cv2.pointPolygonTest(cnt, (x, y), False) >= 0:
                    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, -1)

                    hsv_values = hsv_frame[mask == 255]
                    num_pixels = hsv_values.shape[0]

                    if num_pixels > max_object_size:
                        print(f"Object too large ({num_pixels} pixels), skipping.")
                        return

                    selected_hsv_values.append(hsv_values)

                    # Calculate and save mean HSV
                    mean_hsv = hsv_values.mean(axis=0)
                    with open(output_csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([frame_index] + mean_hsv.round(2).tolist())

                    # Store for green highlight in frame
                    current_click["mask"] = mask
                    current_click["contour"] = cnt
                    print(f"Frame {frame_index} — Selected object with {num_pixels} pixels. Mean HSV: {mean_hsv.round(1)}")
                    break

    # Iterate through video frames
    for frame_index, frame in enumerate(frames):
        fg_mask = back_sub.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        display_frame = frame.copy()

        # Highlight selected object in green
        if current_click["contour"] is not None:
            cv2.drawContours(display_frame, [current_click["contour"]], -1, (0, 255, 0), thickness=cv2.FILLED)

        # Show current frame
        cv2.imshow("Frame", display_frame)
        current_click = {"mask": None, "contour": None}  # reset for next frame
        cv2.setMouseCallback("Frame", click_event, param=(hsv_frame, contours))

        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:  # q or Esc
            print("\nEarly exit — collected data has been saved.")
            break
        elif key == 13:  # Enter
            continue

    cv2.destroyAllWindows()

    if not selected_hsv_values:
        print("No selections were made.")
        return []

    all_hsv = np.concatenate(selected_hsv_values, axis=0)
    print(f"\nCollected a total of {len(all_hsv)} HSV pixels.")
    print(f"HSV averages were written to: {os.path.abspath(output_csv_path)}")

    return all_hsv

if __name__ == '__main__':
    hsv_values = collect_hsv_of_moving_ball(
        "C:\\Users\\elad2\\Downloads\\tryinnn.npy",
        max_object_size=200,
        output_csv_path="C:\\Users\\elad2\\Downloads\\hsv_click_summary.csv"
    )

