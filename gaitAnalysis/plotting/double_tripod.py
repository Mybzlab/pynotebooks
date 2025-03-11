import numpy as np
import sys
import os
import importlib
import glob
import matplotlib.pyplot as plt

# Add the path to the sibling folder to sys.path
sibling_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../3_Testing'))
sys.path.append(sibling_folder_path)

# Import the module using importlib
videoanalyzer = importlib.import_module('videoanalyzer')

def main(): 
    csv_dir = 'Videos/locust/fps_capped/model 3'
    video_dir = 'Videos/locust/fps_capped'

     # Get all .mp4 files in the directory
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

    for video_path, csv_path in zip(video_files, csv_files):
        if video_path.endswith('28.mp4'):
            analyzer = videoanalyzer.VideoAnalyzer(video_path, csv_path)
            brightness_values = analyzer.find_brightness_values()
            brightness_values.pop('head')
            brightness_values.pop('tail')

            tripod_1 = [brightness_values['frontleft'], brightness_values['midright'], brightness_values['backleft']]
            tripod_2 = [brightness_values['frontright'], brightness_values['midleft'], brightness_values['backright']]

            # video 28 specific code:
            # remove first 600 frames
            tripod_1 = [bodypart[450:1000] for bodypart in tripod_1]
            tripod_2 = [bodypart[450:1000] for bodypart in tripod_2]

            brightness_force_conversion = 0.03
            gram_force_multiplier = 0.00980665

            sum_tripod_1 = np.sum(tripod_1, axis=0) * brightness_force_conversion * gram_force_multiplier
            sum_tripod_2 = np.sum(tripod_2, axis=0) * brightness_force_conversion * gram_force_multiplier

            # convert frames to seconds
            start = 0
            stop = (start + len(tripod_1[0])) / analyzer.fps
            step = (stop - start) / len(tripod_1[0])
            plt.plot(np.arange(start, stop, step), sum_tripod_1)
            plt.plot(np.arange(start, stop, step), sum_tripod_2)
            plt.title('Tripods force distribution over time')
            plt.xlabel('Time (s)')
            plt.ylabel('Force (N)')
            plt.legend(['Tripod 1', 'Tripod 2'])
            plt.savefig(f'plotting/tripods/{video_path.split("/")[-1][12:]}.png')
            plt.clf()

if __name__ == "__main__":
    main()