import pandas as pd
import cv2
import math_helpers as mh


class VideoAnalyzer:
    """
    A class to extract data from a video file and a corresponding DeepLabCut CSV file as created by deeplabcut.analyze_videos().
    Extracts all data at once, rather than frame by frame.
    """

    def __init__(self, video_path, csv_path):
        """
        Initialize the VideoAnalyzer object.

        :param video_path: The path to the video file to be analyzed.
        :param csv_path: The path to the corresponding DeepLabCut CSV file.
        """
        self.video_path = video_path
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.bodyparts = list(set(self.df.iloc[0, 1:].tolist()))
        self.cutoff = 0.6

        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self.locations, self.likelihoods = self._extract_data()

    def find_brightness_values(self, half_size=7, mode='gaussian'):
        cap = cv2.VideoCapture(self.video_path)
        brightness_values = {string: [] for string in self.bodyparts}
        for i in range(self.frame_count):
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame {i}.")
                break
            for bodypart in self.bodyparts:
                if self.likelihoods[bodypart][i] > self.cutoff:
                    x, y = self.locations[bodypart][i]
                    if mode == 'gaussian':
                        brightness = mh.gaussian_brightness(frame, x, y, half_size)
                    elif mode == 'max':
                        brightness = mh.max_brightness(frame, x, y, half_size)
                    else:
                        print("Error: Invalid mode.")
                        return
                    brightness_values[bodypart].append(brightness)
                else:
                    brightness_values[bodypart].append(0)
        cap.release()
        return brightness_values
    
    def get_normalized_locations(self):
        coms = self.find_coms()
        normalized_locations = {string: [] for string in self.bodyparts}
        for i in range(len(coms)):
            for bodypart in self.bodyparts:
                if self.likelihoods[bodypart][i] > self.cutoff and self.likelihoods['head'][i] > self.cutoff and self.likelihoods['tail'][i] > self.cutoff:
                    normalized_locations[bodypart].append((self.locations[bodypart][i][0] - coms[i][0], self.locations[bodypart][i][1] - coms[i][1]))
                else:
                    normalized_locations[bodypart].append((None, None))
        return normalized_locations

    def _extract_data(self):
        """
        Extract the brightness values, locations, and likelihoods of the bodyparts from the video file and the DeepLabCut CSV file.

        :param cap: The video capture object.
        """
        locations = {string: [] for string in self.bodyparts}
        likelihoods = {string: [] for string in self.bodyparts}

        for index, _ in self.df[2:].iterrows():
            for bodypart in self.bodyparts:
                # get x, y, and likelihood
                filter = self.df.loc[0] == bodypart
                bodypart_data = self.df.loc[index, filter]
                # grab the x,y of the label
                x = int(float(bodypart_data[0]))
                y = int(float(bodypart_data[1]))
                likelihood = float(bodypart_data[2])

                # add the location of the bodypart to the locations dictionary
                locations[bodypart].append((x, y))

                # add the likelihood of the bodypart to the likelihoods dictionary
                likelihoods[bodypart].append(likelihood)

        return locations, likelihoods

    def _find_com(self, head, tail, fraction):
        """
        Find the center of mass of the insect at a specific time

        :param head: The head of the insect.
        :param tail: The tail of the insect.
        :param fraction: The fraction of the distance between the head and tail where the center of mass is located.

        :return: The center of mass.
        """
        com = (head[0] + fraction * (tail[0] - head[0]),
               head[1] + fraction * (tail[1] - head[1]))
        return com

    def find_coms(self, fraction=0.42):
        """
        Find the centers of mass of the insect

        :param fraction: The fraction of the distance between the head and tail where the center of mass is located.

        :return: A list of the centers of mass.
        """
        coms = []
        for i in range(len(self.locations['head'])):
            head = self.locations['head'][i]
            tail = self.locations['tail'][i]
            com = self._find_com(head, tail, fraction)
            coms.append(com)
        return coms

    def _find_cor(self, past_head, future_head, past_tail, future_tail, tolerance):
        """
        Find the center of rotation of the insect at a specific time

        :param past_head: The head of the insect at a past time.
        :param future_head: The head of the insect at a future time.
        :param past_tail: The tail of the insect at a past time.
        :param future_tail: The tail of the insect at a future time.
        :param tolerance: The tolerance for the denominator of the intersection calculation.
        The lower the tolerance, the more likely the function is to return a center of rotation.

        :return: The center of rotation.
        """
        headx, heady = past_head
        tailx, taily = past_tail
        headx2, heady2, = future_head
        tailx2, taily2 = future_tail
        cor = None

        # Calculate the denominators
        denom = (headx - tailx) * (heady2 - taily2) - \
            (heady - taily) * (headx2 - tailx2)
        if abs(denom) < tolerance:
            return cor  # Lines are parallel or coincident, no cor

        # find the perpendicular bisector of the lines
        ppb1 = mh.find_perpendicular_bisector(
            self.frame_height, (*past_head, *past_tail))
        ppb2 = mh.find_perpendicular_bisector(
            self.frame_height, (*future_head, *future_tail))
        if ppb1 is not None and ppb2 is not None:
            cor = mh.find_intersection(ppb1, ppb2)
        return cor

    def find_cors(self, window=200, tolerance=1000):
        """
        Find the centers of rotation of the insect and the angle of rotation in DEGREES

        :param time_window: The time window for which to capture the rotation in milliseconds. 
        The window is centered around the current frame, capped at the beginning and end of the video.

        :return: A list of the centers of rotation.
        """
        cors = []
        frame_window = (window / 1000) * self.fps
        frame_offset = int(frame_window / 2)
        for i in range(self.frame_count):
            cor = None
            if self.likelihoods['head'][i] > self.cutoff and self.likelihoods['tail'][i] > self.cutoff:
                past_head = self.locations['head'][max(0, i - frame_offset)]
                future_head = self.locations['head'][min(
                    i + frame_offset, self.frame_count - 1)]
                past_tail = self.locations['tail'][max(0, i - frame_offset)]
                future_tail = self.locations['tail'][min(
                    i + frame_offset, self.frame_count - 1)]
                cor = self._find_cor(past_head, future_head,
                                    past_tail, future_tail, tolerance)
            cors.append(cor)

        angles = []
        for i in range(len(cors)):
            if cors[i] is not None:
                head = self.locations['head'][i]
                tail = self.locations['tail'][i]
                angle = mh.calculate_angle(head, cors[i], tail)
                angles.append(angle)
            else:
                angles.append(None)
        return cors, angles

    def normalize_brightness(self, brightness_values):
        """
        Normalize the brightness values of the bodyparts.

        This method adjusts the brightness values of the bodyparts based on their likelihoods and normalizes them.
        Bodyparts with a likelihood below the cutoff are set to 0. The brightness values of bodyparts other than 'head' and 'tail'
        are normalized to a scale of 0 to 100 based on the sum of their brightness values.

        :return: None
        """

        for i in range(self.frame_count):
            brightness_sum = 0
            for bodypart in self.bodyparts:
                if self.likelihoods[bodypart][i] < self.cutoff:
                    brightness_values[bodypart][i] = 0
                if bodypart not in ['head', 'tail']:
                    brightness_sum += brightness_values[bodypart][i]
            if brightness_sum != 0:
                for bodypart in self.bodyparts:
                    if bodypart not in ['head', 'tail']:
                        brightness_values[bodypart][i] = 100 * \
                            brightness_values[bodypart][i] / \
                            brightness_sum
        return brightness_values
    
    def normalize_brightness_over_time(self, brightness_values):
        for bodypart in self.bodyparts:
                brightness_values[bodypart] = [100 * value / max(brightness_values[bodypart]) for value in brightness_values[bodypart]]
        return brightness_values

    def set_unlikely_to_0(self, brightness_values):
        """
        Set the brightness values of bodyparts with a likelihood below the cutoff to 0.

        :return: None
        """
        for i in range(self.frame_count):
            for bodypart in self.bodyparts:
                if self.likelihoods[bodypart][i] < self.cutoff:
                    brightness_values[bodypart][i] = 0
        return brightness_values


def main():
    video_path = r'A:/Uni hdd/Thesis/DLC/Videos/locust/locust.mp4'
    csv_path = r'A:/Uni hdd/Thesis/DLC/Videos/locust/model 2/locustDLC_resnet50_Locust2Sep13shuffle1_100000.csv'

    videoanalyzer = VideoAnalyzer(video_path, csv_path)
    print(videoanalyzer.find_cors())


if __name__ == "__main__":
    main()
