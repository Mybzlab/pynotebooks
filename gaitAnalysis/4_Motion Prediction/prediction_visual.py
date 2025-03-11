import numpy as np
import cv2

# load the predicted directions
directions = np.loadtxt('predicted_directions.csv', delimiter=',', dtype=str)
print(directions)

video_path = r'A:/Uni hdd/Thesis/DLC/Videos/locust/locust.mp4'

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

# Read until video is completed
for index in range(frame_count):

    # Read the frame
    ret, frame = cap.read()
    
    # Check if frame read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Add the predicted direction to the frame at the center of the bottom
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (int(frame_width / 2) - 50, frame_height - 10)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    cv2.putText(frame, directions[index], bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

    # Write the frame into the file 'predicted_video.mp4'
    out.write(frame)
