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
