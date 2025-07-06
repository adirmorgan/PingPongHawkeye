import json
import argparse
import cv2
import numpy as np

def extract_frames_to_npy(config: dict) -> None:
    """
    Extracts frames from a video per configuration and saves them to a NumPy .npy file.

    Config JSON should include:
    {
        "input_video": "path/to/video.mp4",
        "output_npy": "path/to/output.npy",
        "grayscale": true or false,
        "resize": [width, height]
    }
    """
    input_path = config['input_video']
    output_path = config['output_npy']
    grayscale = config.get('grayscale', False)
    resize = config.get('resize', None)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize:
            frame = cv2.resize(frame, tuple(resize))
        frames.append(frame)

    cap.release()
    video_array = np.array(frames)
    np.save(output_path, video_array)
    print(f"Saved {video_array.shape[0]} frames to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames to .npy using JSON config"
    )
    parser.add_argument('config', help='Path to the JSON config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)['mp4_to_npy']

    extract_frames_to_npy(cfg)

if __name__ == '__main__':
    main()
