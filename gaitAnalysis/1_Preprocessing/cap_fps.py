import cv2
import math
import os
import glob

def reduce_fps(input_video_path, output_video_path, target_fps=100):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the input video's properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the nearest divisor of the input FPS that is <= target_fps
    nearest_divisor = min(target_fps, input_fps)
    for i in range(int(input_fps), 0, -1):
        if input_fps % i == 0 and i <= target_fps:
            nearest_divisor = i
            break

    # Calculate the frame step
    frame_step = int(input_fps / nearest_divisor)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_video_path, fourcc, nearest_divisor, (frame_width, frame_height))

    # Read and write frames
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_step == 0:
            out.write(frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    # Directory containing the videos
    video_dir = 'A:/Uni hdd/Thesis/DLC/New Videos/FTIR-TAU/20240811locust5th1491mg1kfps/working folder'
    output_dir = 'A:/Uni hdd/Thesis/DLC/New Videos/FTIR-TAU/20240811locust5th1491mg1kfps/fps_capped'
    target_fps = 100

    # Get all .mp4 files in the directory
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))

    for video_path in video_files:
        video_name = os.path.basename(video_path)
        output_path = os.path.join(output_dir, video_name)

        reduce_fps(video_path, output_path, target_fps)

if __name__ == "__main__":
    main()