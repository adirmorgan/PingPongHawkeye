import json
import cv2
import numpy as np
import argparse


def extract_frames_to_npy(config: dict) -> None:
    """
    Extracts frames from a video per configuration and saves them to a NumPy .npy file.

    Config JSON should include:
    {
        "input_video": "path/to/video.mp4",
        "output_npy": "path/to/output.npy",
        // optional:
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
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total}")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize:
            frame = cv2.resize(frame, tuple(resize))
        frames.append(frame)
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{total}")

    cap.release()
    video_array = np.array(frames)
    print(f"Saving array of shape {video_array.shape} to {output_path}")
    np.save(output_path, video_array)
    print("Frames extraction complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames to .npy using JSON config"
    )
    parser.add_argument(
        'config',
        help='Path to the JSON config file'
    )
    parser.add_argument(
        '--export-config',
        dest='export_config',
        help='Optional path to write back the config JSON'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    config = config['mp4_to_npy']
    extract_frames_to_npy(config)

    if args.export_config:
        with open(args.export_config, 'w') as ef:
            json.dump(config, ef, indent=4)
        print(f"Config exported to {args.export_config}")


if __name__ == '__main__':
    main()
