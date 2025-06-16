import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(coordinates: list, cfg: dict):
    """
    Plot x(t) and y(t) trajectories.
    cfg keys:
      "export_plot"
    """
    x = [pt[0] if pt is not None else None for pt in coordinates]
    y = [pt[1] if pt is not None else None for pt in coordinates]
    t = list(range(len(coordinates)))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title('X vs Frame Index')
    plt.xlabel('Frame')
    plt.ylabel('X')

    plt.subplot(2, 1, 2)
    plt.plot(t, y)
    plt.title('Y vs Frame Index')
    plt.xlabel('Frame')
    plt.ylabel('Y')

    plt.tight_layout()
    export = cfg.get('export_plot', None)
    if export:
        plt.savefig(export)
    plt.show()


def overlay_tracking(frames: np.ndarray, coordinates: list, cfg: dict):
    """
    Display video frames with overlayed tracking points.
    cfg keys:
      "show_window", "display_fps_delay"
    """
    window = cfg.get('show_window', 'Tracking')
    delay = int(cfg.get('display_fps_delay', 30))
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    for idx, frame in enumerate(frames):
        coord = coordinates[idx]
        disp = frame.copy()
        if coord is not None:
            cv2.circle(disp, coord, 5, (0, 255, 0), -1)
        cv2.imshow(window, disp)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Show tracking plots and overlay video using nested JSON config")
    parser.add_argument('config', help='Path to JSON config file')
    parser.add_argument('--export-config', dest='export_config', help='Optional path to re-export full config')
    args = parser.parse_args()

    # load full config and extract 'show' section
    with open(args.config, 'r', encoding='utf-8') as f:
        full_cfg = json.load(f)
    cfg = full_cfg['show']

    # load coordinates
    coords_path = cfg['coordinates_file']
    with open(coords_path, 'r', encoding='utf-8') as cf:
        data = json.load(cf)
    coordinates = data['coordinates']

    # load frames
    frames = np.load(cfg['video_npy'])

    # plot trajectories
    plot_trajectory(coordinates, cfg)

    # overlay tracking on video
    overlay_tracking(frames, coordinates, cfg)

    # optional re-export full config
    if args.export_config:
        with open(args.export_config, 'w', encoding='utf-8') as ef:
            json.dump(full_cfg, ef, indent=4)
        print(f"Config exported to {args.export_config}")

if __name__ == '__main__':
    main()
