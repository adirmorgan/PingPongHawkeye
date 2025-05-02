import cv2
import numpy as np
import os

# Setup video paths
video_dir = r'C:\Users\Bar\Desktop\Itay\Hawkeye\videos'
video_files = ['Shapes1.mp4', 'Shapes2.mp4', 'Bunny&Rabbit.mp4', 'Einstein.mp4']
video_paths = [os.path.join(video_dir, f) for f in video_files]
caps = [cv2.VideoCapture(path) for path in video_paths]

# Determine a common frame size (use the smallest width/height across all videos)
widths  = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  for cap in caps]
heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]
min_w, min_h = min(widths), min(heights)

# Create a resizable window
cv2.namedWindow("Video Grid", cv2.WINDOW_NORMAL)

while True:
    frames, rets = [], []
    # Read and resize each video frame
    for cap in caps:
        ret, frame = cap.read()
        rets.append(ret)
        if ret:
            frame = cv2.resize(frame, (min_w, min_h))
            frames.append(frame)
        else:
            frames.append(np.zeros((min_h, min_w, 3), dtype=np.uint8))
    # Stop if any video ends
    if not all(rets):
        break

    # Build 2×2 grid
    top = np.hstack(frames[0:2])
    bot = np.hstack(frames[2:4])
    grid = np.vstack((top, bot))

    cv2.imshow("Video Grid", grid)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
