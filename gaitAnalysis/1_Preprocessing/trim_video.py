import cv2
import math

def filter_video(video_path, output_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
     
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i >= start_frame and i <= end_frame:
            out.write(frame)

    cap.release()
    out.release()
    
video_path = 'New Videos/FTIR-TAU/20240811locust5th1491mg1kfps/originals/26517_1_29.mp4'
output_path = 'New Videos/FTIR-TAU/20240811locust5th1491mg1kfps/for analysis/26517_1_29_trimmed.mp4'
start_frame = 2000
end_frame = 50000
filter_video(video_path, output_path, start_frame, end_frame)