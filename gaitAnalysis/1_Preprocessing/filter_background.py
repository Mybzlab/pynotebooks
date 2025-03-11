import cv2
import numpy as np
import random
import os
import glob

def calculate_median_frame(video_path, num_frames=50):
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []

    for _ in range(num_frames):
        randomFrameNumber=random.randint(400, total_frames-1)
        # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES,randomFrameNumber)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    return median_frame

def nothing(x):
    pass

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def filter_background(video_path, output_path, median_frame):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    

    # Create a window
    cv2.namedWindow('Adjustments')

    # Create trackbars for alpha, beta and gamma
    cv2.createTrackbar('Contrast', 'Adjustments', 16, 30, nothing)  # Alpha range from 1.0 to 3.0
    cv2.createTrackbar('Brightness', 'Adjustments', 10, 100, nothing)
    cv2.createTrackbar('Gamma', 'Adjustments', 6, 30, nothing) # Scale gamma to 0.1-3.0
    # good values: 20, 10, 10
    
    for i in range(n_frames):
        ret, frame = cap.read()

        if not ret:
            break
        
        # Subtract the median frame from the current frame
        diff_frame = cv2.absdiff(frame, median_frame)

        # Get current positions of the trackbars
        alpha = cv2.getTrackbarPos('Contrast', 'Adjustments') / 2 #10.0  # Scale alpha to 1.0-3.0
        beta = cv2.getTrackbarPos('Brightness', 'Adjustments')
        gamma = cv2.getTrackbarPos('Gamma', 'Adjustments') / 10.0  # Scale gamma to 0.1-3.0

        # Increase contrast
        contrast_frame = cv2.convertScaleAbs(diff_frame, alpha=alpha, beta=beta)

        # Apply gamma correction
        gamma_corrected_frame = adjust_gamma(contrast_frame, gamma=gamma)

        # Create a blank image for the trackbars
        trackbar_img = np.zeros((100, frame_width, 3), dtype=np.uint8)
        
        # Concatenate the video frame and the trackbar image
        combined_frame = cv2.vconcat([gamma_corrected_frame, trackbar_img])

        cv2.imshow('Adjustments', combined_frame)
        out.write(gamma_corrected_frame)
        
        if i % 1000 == 0:
            print(f'Processed {i} frames')

        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Directory containing the videos
video_dir = 'New Videos/FTIR-TAU/20240811locust5th1491mg1kfps/fps_capped'
output_dir = 'New Videos/FTIR-TAU/20240811locust5th1491mg1kfps/high contrast'

# Get all .mp4 files in the directory
video_files = glob.glob(os.path.join(video_dir, '*.mp4'))

for video_path in video_files:
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, video_name)

    # Calculate the median frame
    median_frame = calculate_median_frame(video_path)

    # show median frame
    cv2.imshow('Median Frame', median_frame)

    # Filter the background using the median frame
    filter_background(video_path, output_path, median_frame)