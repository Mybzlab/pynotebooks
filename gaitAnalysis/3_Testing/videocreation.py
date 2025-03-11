import cv2
from painter import Painter
from videoanalyzer import VideoAnalyzer
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk

def user_io():
    # Create a Tkinter root window (it will not be visible)
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Prompt the user to select a file
    print("Select the video you want to analyze")
    video_path = filedialog.askopenfilename(title="video")

    # Ensure a video was selected
    if video_path == "":
        print("No file Selected")
        quit()

    # Print the file path
    print(f"Selected video: {video_path}")

    # Prompt the user to select a file
    print("Select the CSV for this video you want to use")
    csv_path = filedialog.askopenfilename(title="csv")

    # Print the file path
    print(f"Selected CSV: {csv_path}")

    if csv_path == "":
        print("No file Selected")
        quit()
        
    root.destroy()
    
    return video_path, csv_path

def paint_frame(painter, analyzer, displaysettings, displayvalues, index):
    bodyparts = analyzer.bodyparts
    locations = analyzer.locations
    likelihoods = analyzer.likelihoods
    cutoff = analyzer.cutoff
    brightness_values = displayvalues["brightness_values"]
    coms = displayvalues["coms"]
    cors = displayvalues["cors"]
    angles = displayvalues["angles"]

    accepted_bodyparts = [
        bodypart for bodypart in bodyparts if likelihoods[bodypart][index] > cutoff]

    if "head" in accepted_bodyparts and "tail" in accepted_bodyparts:
        if displaysettings["spine"]:
                painter.paint_line(locations["head"][index], locations["tail"][index])

        if displaysettings["com"]:
            painter.paint_com(coms[index])

        if displaysettings["cor"] and cors[index] is not None:
            painter.paint_cor(cors[index])
            painter.paint_line(cors[index], locations["head"][index])
            painter.paint_line(cors[index], locations["tail"][index])
            painter.paint_angle(angles[index], cors[index])

    for bodypart in bodyparts:
        if likelihoods[bodypart][index] > cutoff and bodypart != "head" and bodypart != "tail" and bodypart in accepted_bodyparts:
            if displaysettings["percentagecircle"]:
                painter.paint_percentage(
                    locations[bodypart][index], brightness_values[bodypart][index])
            if displaysettings["percentagetext"]:
                painter.write_percentage(
                    bodypart, locations[bodypart][index], brightness_values[bodypart][index])
            if displaysettings["colorcircle"]:
                painter.paint_circle(locations[bodypart][index], 7, color=(
                    0, 255 - int(2.55*brightness_values[bodypart][index]), int(2.55*brightness_values[bodypart][index])))
            if displaysettings["blackwhitecircle"]:
                painter.paint_circle(locations[bodypart][index], 7, color=(
                    int(2.55*brightness_values[bodypart][index]), int(2.55*brightness_values[bodypart][index]), int(2.55*brightness_values[bodypart][index])))
            if displaysettings["tripodcircles"]:
                if bodypart == "frontleft" or bodypart == "midright" or bodypart == "backleft":
                    painter.paint_circle(locations[bodypart][index], 7, color = (180, 119, 31))
                if bodypart == "frontright" or bodypart == "midleft" or bodypart == "backright":
                    painter.paint_circle(locations[bodypart][index], 7, color= (14, 127, 255))

def selector_GUI(analyzer, brightness_values):
    cap = cv2.VideoCapture(analyzer.video_path)
    frame_count = analyzer.frame_count
    frame_width = analyzer.frame_width
    frame_height = analyzer.frame_height

    likelihoods = analyzer.likelihoods
    cutoff = analyzer.cutoff
    coms = analyzer.find_coms()
    cors, angles = analyzer.find_cors()

    displaysettings = {
        "percentagecircle": False,
        "percentagetext": False,
        "colorcircle": False,
        "blackwhitecircle": False,
        "com": False,
        "cor": False,
        "spine": False,
        "tripodcircles": False
    }

    selected_frame = None
    selected_index = 0

    # select frame where spine and cor are detected
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {i}.")
            break
        if likelihoods["head"][i] > cutoff and likelihoods["tail"][i] > cutoff and cors[i] is not None and cors[i][0] > 0 and cors[i][1] > 0 and cors[i][0] < frame_width and cors[i][1] < frame_height:
            selected_frame = frame
            selected_index = i
            break

        displayvalues = {
        "brightness_values": brightness_values,
        "coms": coms,
        "cors": cors,
        "angles": angles
        }

    if selected_frame is None:
        print("Error: Found no frame suitable for preview.")
        return

    # Function to handle button clicks
    def toggle(settings_key):
        displaysettings[settings_key] = not displaysettings[settings_key]
        update_frame()

    def quit_selection():
        root.destroy()
        cap.release()
        cv2.destroyAllWindows()

    def update_frame():
        painter = Painter(selected_frame.copy())
        paint_frame(painter, analyzer, displaysettings, displayvalues, selected_index)
        updated_frame = painter.frame

        frame_rgb = cv2.cvtColor(updated_frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        frame_photo = ImageTk.PhotoImage(image=frame_image)
        frame_label.config(image=frame_photo)
        frame_label.image = frame_photo

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Preview")

    # Convert the selected frame to an image that Tkinter can display
    frame_rgb = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)
    frame_image = Image.fromarray(frame_rgb)
    frame_photo = ImageTk.PhotoImage(image=frame_image)

    # Create a label to display the frame
    frame_label = tk.Label(root, image=frame_photo)
    frame_label.pack()

    # Create buttons
    for key in displaysettings:
        button = tk.Button(root, text=f"Toggle {key.capitalize()}",
                           command=lambda key=key: toggle(key))
        button.pack(side=tk.LEFT, padx=10, pady=10)
        
    # spine_button = tk.Button(root, text="Toggle Spine", command=toggle_spine)
    # spine_button.pack(side=tk.LEFT, padx=10, pady=10)

    quit_button = tk.Button(root, text="Save & Generate", bg='cornflower blue', command=quit_selection)
    quit_button.pack(side=tk.LEFT, padx=10, pady=10)

    # Start the Tkinter event loop
    root.mainloop()

    return displaysettings, displayvalues


def create_video(analyzer, displaysettings, displayvalues):
    # TODO VERY IMPORTANT CHANGE THIS BACK AFTER PRESENTATION
    cap = cv2.VideoCapture(r'A:/Uni hdd/Thesis/DLC/Videos/locust/fps_capped/high contrast/26517_1_28.mp4')

    # cap = cv2.VideoCapture(analyzer.video_path)
    # Get video properties
    csv_path = analyzer.csv_path
    fps = analyzer.fps
    frame_width = analyzer.frame_width
    frame_height = analyzer.frame_height
    frame_count = analyzer.frame_count

    # output video path. It is stored in the same folder as the csv path
    output_video_path = csv_path.rpartition('/')[0] + '/painting.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps,
                          (frame_width, frame_height))

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {i}.")
            break

        # filter bodyparts below the detection threshold
        painter = Painter(frame)
        paint_frame(painter, analyzer, displaysettings, displayvalues, i)

        # write the painted frame to the output video
        painted_frame = painter.frame
        out.write(painted_frame)

    cap.release()
    out.release()


def main():
    video_path = r'A:/Uni hdd/Thesis/DLC/Videos/locust/fps_capped/26517_1_28.mp4'
    csv_path = r'A:/Uni hdd/Thesis/DLC/Videos/locust/fps_capped/model 3/26517_1_28DLC_resnet50_More LocustOct25shuffle1_150000.csv'
    
    # (optional) let the user select the video and csv
    # video_path, csv_path = user_io()

    analyzer = VideoAnalyzer(video_path, csv_path)
    brightness_values = analyzer.find_brightness_values(mode='gaussian')
    # brightness_values = analyzer.normalize_brightness(brightness_values)
    brightness_values = analyzer.normalize_brightness_over_time(brightness_values)
    
    displaysettings, displayvalues = selector_GUI(analyzer, brightness_values)
    create_video(analyzer, displaysettings, displayvalues)


if __name__ == "__main__":
    main()
