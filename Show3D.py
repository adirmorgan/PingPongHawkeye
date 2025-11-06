import json
import argparse
import matplotlib.pyplot as plt

def plot_trajectory(coords, config):
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
    export_path = config.get('export_plot')
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

    # load coordinates
    coords_file = show_cfg.get('coordinates_file')
    with open(coords_file, 'r', encoding='utf-8') as jf:
        coords = json.load(jf)

    # plot trajectory
    plot_trajectory(coords, show_cfg)

    # optional re-export config
    if args.export_config:
        with open(args.export_config, 'w', encoding='utf-8') as ef:
            json.dump(full_cfg, ef, indent=4)
        print(f"Config exported to {args.export_config}")

if __name__ == '__main__':
    main()
