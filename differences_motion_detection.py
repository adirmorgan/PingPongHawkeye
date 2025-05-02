import cv2
import numpy as np
import os

# --- Parameters ---
video_path = r'C:\Users\Bar\Desktop\Itay\Hawkeye\videos\Shapes1.mp4'
k = 5           # number of frames back to compare
thresh = 25     # motion threshold
window_name = f"{k}-Step Motion Detection"

# --- Setup ---
cap = cv2.VideoCapture(video_path)
frame_buffer = []

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_buffer.append(gray)

    if len(frame_buffer) > k:
        # keep the buffer with size k!
        frame_buffer.pop(0)

    if len(frame_buffer) < k:
        # buffer not fully filled: show blank or waiting frame
        motion_mask = np.zeros_like(gray)

    else:
        # compare current frame with the one k steps ago
        kstep_diff = cv2.absdiff(frame_buffer[0], gray)
        _, motion_mask = cv2.threshold(kstep_diff, thresh, 255, cv2.THRESH_BINARY)

    cv2.imshow(window_name, motion_mask)
