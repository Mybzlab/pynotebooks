from videoanalyzer import VideoAnalyzer
import matplotlib.pyplot as plt
import math
import math_helpers as mh
import numpy as np
import cv2
from painter import Painter

def curvity(i, m, d):
    return (m * d) / (i + m * (d**2))

def main():
    # video_path = r'A:/Uni hdd/Thesis/DLC/Videos/locust/fps_capped/locust.mp4'
    # csv_path = r'A:/Uni hdd/Thesis/DLC/Videos/locust/fps_capped/model 3/locustDLC_resnet50_More LocustOct25shuffle1_150000.csv'

    video_path = r'A:/Uni hdd/Thesis/DLC/Videos/crazyant/crazyant.mp4'
    csv_path = r'A:/Uni hdd/Thesis/DLC/Videos/crazyant/various model/crazyantDLC_resnet50_VariousOct2shuffle1_50000.csv'
    analyzer = VideoAnalyzer(video_path, csv_path)
    normalized_locations = analyzer.get_normalized_locations()

    # locust properties
    m = 1.5 # mass in grams
    i = 3 # moment of inertia in g*cm^2
    l = 4.5 # length in cm

    # fireant properties
    m = 0.005 # mass in grams
    i = 0.0001 # moment of inertia in g*cm^2
    l = 0.5 # length in cm

    deltas = {string: [] for string in analyzer.bodyparts}
    # window_size = 10 # window size in frames
    for bodypart in analyzer.bodyparts:
        plt.title(f'{bodypart} delta')
        plt.plot([normalized_locations[bodypart][i][0] for i in range(len(normalized_locations[bodypart]))])
        plt.plot([normalized_locations[bodypart][i][1] for i in range(len(normalized_locations[bodypart]))])
        plt.legend(['dx', 'dy'])
        plt.xlabel("Time")
        plt.ylabel("Delta")
        plt.plot(deltas[bodypart])
        plt.savefig(f"../DLC/plotting/deltas/{bodypart}_delta.png")
        plt.clf()

    # none to 0 for kappa calculations
    deltas = {string: [] for string in analyzer.bodyparts}
    kappas = {string: [] for string in analyzer.bodyparts}

    rotated_normalized_locations = {string: [] for string in analyzer.bodyparts}
    for i in range(len(normalized_locations["head"])):
        for bodypart in analyzer.bodyparts:
            vector = normalized_locations[bodypart][i]
            if vector == (None, None):
                rotated_normalized_locations[bodypart].append((np.nan, np.nan)) # np.nan instead of None
                deltas[bodypart].append(np.nan)   # changes here
                # deltas[bodypart].append(0)
            else:
                rot_vector = mh.rotate_to_align_with_y_axis(vector, normalized_locations["tail"][i])
                rotated_normalized_locations[bodypart].append(rot_vector)
                deltas[bodypart].append(rot_vector[1])

    # TODO: detect first value that isn't None instead of taking a specific value
    conversion_rate = l / (rotated_normalized_locations["tail"][300][1] - rotated_normalized_locations["head"][300][1]) # cm per pixel
    # print(conversion_rate) 


    for bodypart in analyzer.bodyparts:
        # plot trajectory of bodyparts
        plt.title(f'{bodypart} delta')
        plt.plot([rotated_normalized_locations[bodypart][i][0] for i in range(len(rotated_normalized_locations[bodypart]))])
        plt.plot([rotated_normalized_locations[bodypart][i][1] for i in range(len(rotated_normalized_locations[bodypart]))])
        plt.legend(['dx', 'dy'])
        plt.xlabel("Time")
        plt.ylabel("Delta")
        plt.savefig(f"../DLC/plotting/rotated_deltas/{bodypart}_delta.png")
        plt.clf()

        # extract deltas and convert to cm
        deltas[bodypart] = np.array(deltas[bodypart]) * conversion_rate
        kappas[bodypart] = curvity(1, 1, deltas[bodypart])

        # plot kappas
        plt.title(f'{bodypart} bodypart contribution to curvity')
        plt.plot(kappas[bodypart])

        plt.legend(['kappa'])
        plt.xlabel("Time")
        plt.ylabel("Kappa")
        plt.savefig(f"../DLC/plotting/rotated_deltas/{bodypart}_kappa.png")
        plt.clf()


    # sum all kappas
    total_kappa = np.zeros(len(kappas["head"]))
    for bodypart in analyzer.bodyparts:
        if bodypart != "head" and bodypart != "tail":
            total_kappa += np.nan_to_num(kappas[bodypart])

    # divide total_kappa elementwise by the number of bodyparts that arent np.nan
    for i in range(len(total_kappa)):
        counter = 0
        for bodypart in analyzer.bodyparts:
            if bodypart != "head" and bodypart != "tail" and np.isnan(kappas[bodypart][i]):
                counter += 1
        total_kappa[i] = total_kappa[i] / (len(analyzer.bodyparts) - 2 - counter)

    plt.title(f'Total curvity')

    # plot x-axis
    plt.plot([0 for i in range(len(total_kappa))], color='black')

    plt.plot(total_kappa)
    plt.legend(['kappa'])
    plt.xlabel("Time")
    plt.ylabel("Kappa")
    plt.savefig(f"../DLC/plotting/rotated_deltas/total_kappa_{video_path[-6:-4]}.png")
    plt.clf()


    # print(deltas["head"])
    # print(deltas['frontleft'])

    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object to save rotated video
    output_size = 250  # Square size (e.g., 200x200)
    out = cv2.VideoWriter('../DLC/plotting/rotated_deltas/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_size, output_size))

    # Rotation angle
    coms = analyzer.find_coms()

    rotation_angle = 0

    for i in range(analyzer.frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        if normalized_locations['head'][i] != (None, None):
            rotation_angle = mh.get_rot_angle_y_axis(normalized_locations['tail'][i])  # Change to desired angle
            rotation_angle = np.rad2deg(rotation_angle)
            painter = Painter(frame)
            # painter.paint_line(coms[i], analyzer.locations['head'][i])
            # painter.paint_line(analyzer.locations['head'][i], analyzer.locations['tail'][i])
            # midline = mh.find_perpendicular_bisector(frame_height, (analyzer.locations['head'][i][0], analyzer.locations['head'][i][1], analyzer.locations['tail'][i][0], analyzer.locations['tail'][i][1]))
            # painter.paint_line((midline[0], midline[1]), (midline[2], midline[3]))
            painter.paint_com(coms[i])
            # for bodypart in analyzer.bodyparts:
            #     if normalized_locations[bodypart][i] != (None, None):
            #         painter.paint_circle(analyzer.locations[bodypart][i], 5)

        # Get the center of the frame
        center = coms[i]

        # Compute the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)

        # Get the rotated frame
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame_width, frame_height))

        # Define the square region of interest (ROI) centered at the COM
        center_x, center_y = int(coms[i][0]), int(coms[i][1])

        # Calculate the top-left and bottom-right corners of the square
        half_size = output_size // 2
        top_left_x = center_x - half_size
        top_left_y = center_y - half_size
        bottom_right_x = center_x + half_size
        bottom_right_y = center_y + half_size

        # Create a black canvas for the output square
        square_roi = np.zeros((output_size, output_size, 3), dtype=np.uint8)

        # Determine valid frame coordinates
        valid_top_left_x = max(top_left_x, 0)
        valid_top_left_y = max(top_left_y, 0)
        valid_bottom_right_x = min(bottom_right_x, frame_width)
        valid_bottom_right_y = min(bottom_right_y, frame_height)

        # Determine where to place the valid part on the black canvas
        canvas_top_left_x = max(0, -top_left_x)
        canvas_top_left_y = max(0, -top_left_y)

        # Extract the valid ROI from the frame
        valid_roi = rotated_frame[valid_top_left_y:valid_bottom_right_y, valid_top_left_x:valid_bottom_right_x]

        # Place the valid ROI onto the black canvas
        square_roi[
            canvas_top_left_y:canvas_top_left_y + valid_roi.shape[0],
            canvas_top_left_x:canvas_top_left_x + valid_roi.shape[1]
        ] = valid_roi

        window_size = analyzer.frame_count
        window_size = 1

        # paint the bodypart trail and the delta arrow
        painter = Painter(square_roi)
        for bodypart in analyzer.bodyparts:
            if bodypart != "head" and bodypart != "tail":
                for j in range(max(0, i - window_size), i):
                    if normalized_locations[bodypart][j] != (None, None):
                        location = rotated_normalized_locations[bodypart][j].copy()
                        location[0] = location[0] + half_size
                        location[1] = location[1] + half_size
                        # painter.paint_circle(location, 1, color=(0, 0, int(255-(((i-j)/window_size)*255))))
                        start = (location[0], half_size)
                        end = location
                        if location[1] > half_size:
                            color = (0, 0, 255)
                        else:
                            color = (0, 255, 0)
                        painter.paint_arrow(start, end, color)

        # Write the padded square ROI to the output video
        out.write(square_roi)

        # Display the frame (optional)
        cv2.imshow('Rotated Frame', square_roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    main()