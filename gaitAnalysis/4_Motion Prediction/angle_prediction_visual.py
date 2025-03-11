import numpy as np
import cv2
import importlib
import sys
import os

# Add the path to the sibling folder to sys.path
sibling_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../3_Testing'))
sys.path.append(sibling_folder_path)
# Import the module using importlib
videoanalyzer = importlib.import_module('videoanalyzer')
ptr = importlib.import_module('painter')
# mh = importlib.import_module('mathhelpers')



# load the predicted directions
predicted_angles = np.loadtxt('predicted_angles.csv', delimiter=',', dtype=str)
real_angles = np.loadtxt('actual_angles.csv', delimiter=',', dtype=str)

video_path = r'A:/Uni hdd/Thesis/DLC/Videos/locust/fps_capped/locust.mp4'
csv_path = r'A:/Uni hdd/Thesis/DLC/Videos/locust/fps_capped/model 3/locustDLC_resnet50_More LocustOct25shuffle1_150000.csv'

analyzer = videoanalyzer.VideoAnalyzer(video_path, csv_path)
head_locs = analyzer.locations['head']
tail_locs = analyzer.locations['tail']
locations = analyzer.locations
cors, angles = analyzer.find_cors()

# load a video, add the predicted directions to the video and save it
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video's frame width, height, and frames per second
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))
frame_count = int(cap.get(7))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter('Videos/locust/prediction/predicted_video.mp4', fourcc, fps, (frame_width, frame_height))

# # TODO automate this
# frames2cut = 12

# # remove frames2cut frames from the end of X and the beginning of y. X and y pairs should now be separated by (time_window) amount of time
# real_angles = real_angles[:len(real_angles)-frames2cut]
# predicted_angles = predicted_angles[frames2cut:]
# predicted_angles = np.append(predicted_angles, [0] * frames2cut)
# real_angles = np.append(real_angles, [0] * frames2cut)
# print(real_angles)

# Read until video is completed
print(frame_count)
for index in range(frame_count-frames2cut):

    # Read the frame
    ret, frame = cap.read()
    
    # Check if frame read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Calculate the arrow's end point based on the angle
    if real_angles[index] != 'nan' and predicted_angles[index] != 'nan':
        painter = ptr.Painter(frame)

        # paint spine
        painter.paint_line(head_locs[index], tail_locs[index], (255, 255, 255))
        # Calculate the angle between the line (head_locs[index], tail_locs[index]) and a horizontal line
        dx = tail_locs[index][0] - head_locs[index][0]
        dy = tail_locs[index][1] - head_locs[index][1]
        base_angle = 180 - np.degrees(np.arctan2(dy, dx))
        angle = base_angle + float(real_angles[index])

        # arrow specifications
        length = 50
        center_x, center_y = head_locs[index]
        end_x = int(center_x + length * np.cos(np.radians(angle)))
        end_y = int(center_y - length * np.sin(np.radians(angle)))

        # Draw the arrow on the frame
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (255, 255, 255), 3, tipLength=0.3)

        angle = base_angle + float(predicted_angles[index])
        # other arrow specifications
        length = 40  # Length of the arrow
        center_x, center_y = head_locs[index]
        end_x = int(center_x + length * np.cos(np.radians(angle)))
        end_y = int(center_y - length * np.sin(np.radians(angle)))

        # Draw the arrow on the frame
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3, tipLength=0.3)
        # head = head_locs[index]
        # tail = tail_locs[index]
        # cor = cors[index]
        # if cor is not None:
        #     painter.paint_cor(cor, (255, 255, 255))
        #     painter.paint_line(head, cor, (255, 255, 255))
        #     painter.paint_line(tail, cor, (255, 255, 255))

        # TODO add the predicted cor to the video. find out how to caluculate predicted cor from predicted angle

        
        

    # Write the frame into the file 'predicted_video.mp4'
    out.write(frame)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows() 
