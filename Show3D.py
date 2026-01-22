import json
import argparse
import matplotlib.pyplot as plt
from utils import *

def plot_trajectory(coords,  export_path):
    """
    Plot x, y, z vs time and save to file.
    """
    t = [pt.get('t') for pt in coords]
    x = [pt.get('x') for pt in coords]
    y = [pt.get('y') for pt in coords]
    z = [pt.get('z') for pt in coords]

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    axes[0].plot(t, x)
    axes[0].set_ylabel('X')
    axes[0].set_title('X vs Time')

    axes[1].plot(t, y)
    axes[1].set_ylabel('Y')
    axes[1].set_title('Y vs Time')

    axes[2].plot(t, z)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Z')
    axes[2].set_title('Z vs Time')

    plt.tight_layout()
    if export_path:
        plt.savefig(export_path)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot 3D trajectory from coordinates JSON")
    parser.add_argument('config', help='Path to JSON config file')
    parser.add_argument('--export-config', dest='export_config', help='Optional path to re-export config')
    args = parser.parse_args()

    # load config
    with open(args.config, 'r', encoding='utf-8') as cf:
        full_cfg = json.load(cf)
    show_cfg = full_cfg.get('show', {})

    # check configuration
    if not show_cfg:
        raise ValueError("No 'show' section found in config")
    if not show_cfg.get('export_plots'):
        raise ValueError("No 'export_plots' found in 'show' section")
    if not show_cfg.get('import_coords'):
        raise ValueError("No 'coordinates_file' found in 'show' section")
    if len(show_cfg['export_plots'])!=len(show_cfg['import_coords']):
        raise ValueError("Number of 'export_plots' does not match number of 'coordinates_file'")

    # plot trajectory
    for i,export_path in enumerate(show_cfg['export_plots']):
        # load coordinates
        coords_file = show_cfg['import_coords'][i]
        with open(coords_file, 'r', encoding='utf-8') as jf:
            coords = json.load(jf)
        plot_trajectory(coords, export_path)

    # optional re-export config
    if args.export_config:
        with open(args.export_config, 'w', encoding='utf-8') as ef:
            json.dump(full_cfg, ef, indent=4)
        print(f"Config exported to {args.export_config}")

if __name__ == '__main__':
    main()
