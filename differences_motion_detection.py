import cv2
import numpy as np
import os

# --- Parameters ---
video_path = r'C:\Users\Bar\Desktop\Itay\Hawkeye\videos\Shapes1.mp4'
k = 5           # number of frames back to compare
thresh = 25     # motion threshold
window_name = f"{k}-Step Motion Detection"
