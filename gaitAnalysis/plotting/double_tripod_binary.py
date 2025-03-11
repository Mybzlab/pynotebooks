import os
import sys
import glob
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add the path to the sibling folder to sys.path
sibling_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../3_Testing'))
sys.path.append(sibling_folder_path)

# Import the module using importlib
videoanalyzer = importlib.import_module('videoanalyzer')

def find_time_windows(binary_values):
    time_windows = {}
    for key, values in binary_values.items():
        windows = []
        start = None
        for i, value in enumerate(values):
            if value == 1 and start is None:
                start = i
            elif value == 0 and start is not None:
                windows.append((start, i - start))
                start = None
        if start is not None:
            windows.append((start, len(values) - start))
        time_windows[key] = windows
    return time_windows

def main(): 
    csv_dir = 'Videos/locust/fps_capped/model 3'
    video_dir = 'Videos/locust/fps_capped'

    # Get all .mp4 files in the directory
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

    for video_path, csv_path in zip(video_files, csv_files):
        if not video_path.endswith('28.mp4'):
            continue
        print(video_path)
        analyzer = videoanalyzer.VideoAnalyzer(video_path, csv_path)
        brightness_values = analyzer.find_brightness_values()
        brightness_values.pop('head')
        brightness_values.pop('tail')
        brightness_values = {key: values[550:1000] for key, values in brightness_values.items()}
        # Convert brightness values to binary (1 if > 0, else 0)
        binary_values = {key: np.where(np.array(values) > 5, 1, 0) for key, values in brightness_values.items()}
        
        # Find time windows
        time_windows = find_time_windows(binary_values)

        # Convert time windows from frames to seconds
        for key, values in time_windows.items():
            for i, value in enumerate(values):
                time_windows[key][i] = (value[0] / analyzer.fps, value[1] / analyzer.fps)

        # Desired order of keys
        ordered_keys = ['frontleft', 'midleft', 'backleft', 'frontright', 'midright', 'backright']

        # Normalize brightness values
        norm = mcolors.Normalize(vmin=0, vmax=5)
        cmap1 = plt.get_cmap('Oranges')
        cmap2 = plt.get_cmap('Blues')

        # Plot the binary values as black bars, each key on its own line
        plt.figure()
        offset = -0.5

        start = 0
        stop = (start + len(brightness_values["frontright"])) / analyzer.fps
        step = (stop - start) / len(brightness_values["frontright"])

        for key in ordered_keys:
            if key in time_windows:
                values = time_windows[key]
                color = "(0,0,0)"
                if key == 'frontleft' or key == 'midright' or key == 'backleft':
                    color = "tab:blue"
                elif key == 'frontright' or key == 'midleft' or key == 'backright':
                    color = "tab:orange"
                plt.broken_barh(values, (offset+0.1, 0.9), facecolors=color)
                offset += 1

        # for key in ordered_keys:
        #     if key in time_windows:
        #         values = time_windows[key]
        #         for (start, duration) in values:
        #             color = cmap1(norm(brightness_values[key][int(start * analyzer.fps)]))
        #             plt.broken_barh([(start, duration)], (offset + 0.1, 0.9), facecolors=color)
        #         offset += 1

        plt.xlabel('Time (s)')
        plt.ylabel('Leg')
        plt.title('Contact over time')
        plt.yticks(np.arange(len(ordered_keys)), ordered_keys)  # Set y-ticks to the ordered keys
        plt.savefig(f'plotting/binary_tripods/{video_path.split("/")[-1][12:]}.png')
        plt.clf()

if __name__ == "__main__":
    main()