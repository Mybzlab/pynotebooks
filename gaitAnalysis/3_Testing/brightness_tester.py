from videoanalyzer import VideoAnalyzer
import numpy as np
import matplotlib.pyplot as plt

def main():
    video_path = r'A:/Uni hdd/Thesis/DLC/Videos/locust/fps_capped/26517_1_25.mp4'
    csv_path = r'A:/Uni hdd/Thesis/DLC/Videos/locust/fps_capped/model 3/26517_1_25DLC_resnet50_More LocustOct25shuffle1_150000.csv'

    # video_path = r'A:/Uni hdd/Thesis/DLC/Videos/locust/fps_capped/locust.mp4'
    # csv_path = r'A:/Uni hdd/Thesis/DLC/Videos/locust/fps_capped/model 3/locustDLC_resnet50_More LocustOct25shuffle1_150000.csv'

    analyzer = VideoAnalyzer(video_path, csv_path)
    modes = ['gaussian', 'max']
    half_size = 7

    for mode in modes:
        brightness_values = analyzer.find_brightness_values(half_size, mode)
        brightness_values = analyzer.set_unlikely_to_0(brightness_values)

        # sum all brightness values over the index except the head and tail
        sum = [np.sum([brightness_values[bodypart][i] for bodypart in brightness_values if bodypart not in ['head', 'tail']]) for i in range(analyzer.frame_count)]

        # plot the sum of the brightness values
        plt.plot(sum)
        plt.title(f'Sum of {mode} brightness values over time')
        plt.xlabel('Frame')
        plt.ylabel('Sum of brightness values')
        plt.ylim(bottom=0)
        plt.show()

if __name__ == "__main__":
    main()