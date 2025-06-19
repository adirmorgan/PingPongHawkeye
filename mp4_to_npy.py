import json
import cv2
import numpy as np
import argparse


def extract_frames_to_npy(config: dict) -> None:
    """
    Extracts frames from a video per configuration and saves them to a NumPy .npy file.
    
    Config JSON should include:
    {
        "input_video": "path/to/video.mp4",  # Path to the input video file.
        "output_npy": "path/to/output.npy", # Path to save the output .npy file with extracted frames.
        // optional:
        "grayscale": true or false,         # Convert frames to grayscale if true (default: false).
        "resize": [width, height]           # Resize frames to the provided dimensions [width, height].
    }
    """
    input_path = config['input_video']
    output_path = config['output_npy']
    grayscale = config.get('grayscale', False)
    resize = config.get('resize', None)

    # Open the video file using OpenCV to read frames
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        # If the video cannot be opened, raise an error
        raise IOError(f"Cannot open video file: {input_path}")

    # Initialize a list to store the extracted and processed frames
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
    print(f"Total frames: {total}")  # Log the total frame count

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if grayscale:
            # Convert the frame to grayscale if the 'grayscale' option is enabled
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize:
            # Resize the frame to the specified dimensions if the 'resize' option is provided
            frame = cv2.resize(frame, tuple(resize))
        frames.append(frame)  # Add the processed frame to the list
        count += 1
        if count % 100 == 0:
            # Log progress every 100 frames
            print(f"Processed {count}/{total}")

    cap.release()
    # Convert the list of frames into a single NumPy array
    video_array = np.array(frames)
    print(f"Saving array of shape {video_array.shape} to {output_path}")  # Log the shape of the array
    
    # Save the frames as a NumPy .npy file to the specified output location
    np.save(output_path, video_array)
    print("Frames extraction complete.")


def main():
    # Main function that handles argument parsing and starts the frame extraction process
    parser = argparse.ArgumentParser(
        description="Extract frames to .npy using JSON config"
    )
    
    # Argument for specifying the path to the JSON configuration file (required)
    parser.add_argument(
        'config',
        help='Path to the JSON config file'
    )
    
    # Optional argument to export the updated JSON configuration after processing
    parser.add_argument(
        '--export-config',
        dest='export_config',
        help='Optional path to write back the config JSON'
    )
    args = parser.parse_args()

    # Load the configuration JSON file specified by the user
    with open(args.config, 'r') as f:
        config = json.load(f)
    config = config['mp4_to_npy']  # Extract the specific 'mp4_to_npy' part of the configuration
    
    # Process the video according to the provided configuration
    extract_frames_to_npy(config)
    
    # If the optional --export-config argument is provided, save the config to a new file
    if args.export_config:
        with open(args.export_config, 'w') as ef:
            json.dump(config, ef, indent=4)
        print(f"Config exported to {args.export_config}")


if __name__ == '__main__':
    main()
