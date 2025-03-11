import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.layers import Dense
from keras.models import Sequential
import importlib
import sys
import os
import glob
import matplotlib.pyplot as plt

# Add the path to the sibling folder to sys.path
sibling_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../3_Testing'))
sys.path.append(sibling_folder_path)

# Import the module using importlib
videoanalyzer = importlib.import_module('videoanalyzer')

def train_classifier(X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
    """
    Train a neural network with 6 inputs, 1 output representing an angle, and 2 fully connected hidden layers.
    
    Parameters:
    X_train (numpy.ndarray): Training data with shape (num_samples, 6)
    y_train (numpy.ndarray): Training labels with shape (num_samples, 1)
    X_val (numpy.ndarray): Validation data with shape (num_samples, 6)
    y_val (numpy.ndarray): Validation labels with shape (num_samples, 1)
    epochs (int): Number of epochs to train the model
    batch_size (int): Batch size for training
    
    Returns:
    model (tf.keras.Model): Trained neural network model
    history (tf.keras.callbacks.History): Training history
    """
    # Create a Sequential model 
    model = Sequential()
    
    # Add input layer with 6 inputs
    model.add(Dense(16, input_dim=6, activation='relu'))
    
    # Add first fully connected hidden layer
    model.add(Dense(16, activation='relu'))
    
    # Add second fully connected hidden layer
    model.add(Dense(16, activation='relu'))
    
    # Add output layer with 1 output
    model.add(Dense(1, activation='tanh'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    
    return model, history

def run_model(time_window):
    # Initialize empty lists to collect all X and y values
    all_X = []
    all_y = []
    visual_X = []
    visual_y = []
    visual_likelihoods = []
    X_test = []
    y_test = []

    # Directories containing the videos and corresponding csvs
    csv_dir = 'Videos/locust/fps_capped/model 3'
    video_dir = 'Videos/locust/fps_capped'
    chosen_video_path = 'Videos/locust/fps_capped/locust.mp4'
    chosen_video_end = chosen_video_path.split('/')[-1]

    # # time window to predict the angle in ms so 200 means predicting 200 ms into the future. 
    # Not the same as the time window used to calculate the angle
    # time_window = 200

    # Get all .mp4 files in the directory
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

    for video_path, csv_path in zip(video_files, csv_files):

        # remove video 26 and 27 for now
        # TODO: remove after model improvement
        if video_path.endswith('26.mp4') or video_path.endswith('27.mp4'):# or video_path.endswith('locust.mp4'):
            print(f"Skipping video {video_path}")
            continue

        if not video_path.endswith(chosen_video_end):# or video_path.endswith('25.mp4'):
            continue

        # brightness_values, directions, angles = av.analyze_data(video_path, csv_path)
        analyzer = videoanalyzer.VideoAnalyzer(video_path, csv_path)
        brightness_values = analyzer.find_brightness_values(mode='gaussian')
        brightness_values = analyzer.set_unlikely_to_0(brightness_values)
        brightness_values = analyzer.normalize_brightness(brightness_values)
        # brightness_values = analyzer.normalize_brightness(brightness_values)
        likelihoods = analyzer.likelihoods
        _, angles = analyzer.find_cors(tolerance=1000)

        # extract features from brightness_values and directions
        X = np.array([brightness_values["frontright"], brightness_values["frontleft"], brightness_values["midright"], brightness_values["midleft"], brightness_values["backright"], brightness_values["backleft"]])

        # flip axes to have shape (num_samples, 6)
        X = np.transpose(X)

        # print(y_train)
        y = np.array(angles)

        # replace all None values in y with 0
        y = np.array([0 if v is None else v for v in y])
        # y = np.nan_to_num(y)

        # scale y between -1 and 1
        y = y/180

        #TODO before or after frames2cut? After, but requires altering frames2cut in visual file
        if video_path.endswith(chosen_video_end):
            visual_X = X
            visual_y = y
            visual_likelihoods = likelihoods

        # calculate the number of frames to cut from the beginning and end of the video
        frames2cut = int(analyzer.fps * time_window / 1000)

        # remove frames2cut frames from the end of X and the beginning of y. X and y pairs should now be separated by (time_window) amount of time
        X = X[:analyzer.frame_count-frames2cut]
        y = y[frames2cut:]

        # cut the likelihoods the same way as X to keep corresponding values
        likelihoods = {k: v[:analyzer.frame_count-frames2cut] for k, v in likelihoods.items()}

        # remove all entries where likelihood["head"] < 0.6 and likelihood["tail"] < 0.6
        indices = np.where(np.logical_and(np.array(likelihoods["head"]) > 0.6, np.array(likelihoods["tail"]) > 0.6))
        X = X[indices]
        y = y[indices]

        if video_path.endswith(chosen_video_end):
            X_test = X
            y_test = y
        # else:
        # Append the current X and y values to the lists
            all_X.append(X)
            all_y.append(y)


    # Combine all X and y values into one big array
    all_X = np.vstack(all_X)
    all_y = np.hstack(all_y)
    
    # # Remove 90% of entries where y is 0
    # zero_indices = np.where(all_y == 0)[0]
    # non_zero_indices = np.where(all_y != 0)[0]
    
    # # Randomly select 10% of zero indices to keep
    # keep_zero_indices = np.random.choice(zero_indices, size=int(0.1 * len(zero_indices)), replace=False)
    
    # # Combine the indices to keep
    # keep_indices = np.concatenate((non_zero_indices, keep_zero_indices))
    
    # # Filter all_X and all_y to keep only the selected indices
    # all_X = all_X[keep_indices]
    # all_y = all_y[keep_indices]

    # plot a histogram of all values in y
    plt.hist(all_y, bins=50)
    plt.show()

    # create train test split
    # X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2)

    X_train = all_X
    y_train = all_y

    # Example usage (assuming X_train and y_train are already defined)
    model, history = train_classifier(X_train, y_train, X_test, y_test)

    # evaluate the model
    loss, mae = model.evaluate(X_test, y_test)

    print(f'Loss: {loss}, MAE: {mae}')

    predicted_angles = model.predict(X_test) * 180
    actual_angles = y_test * 180

    # for calculating RMSE, remove all enties where actual_angles == 0
    # indices = np.where(actual_angles != 0)
    # predicted_angles = predicted_angles[indices]
    # actual_angles = actual_angles[indices]

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual_angles, predicted_angles))
    print(f'RMSE: {rmse}')

    # Calculate R-squared
    r2 = r2_score(actual_angles, predicted_angles)
    print(f'R-squared: {r2}')

    # Plot training and validation loss over epochs
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.show()


    # generate csv files for visualisation
    predicted_angles = model.predict(visual_X) * 180
    actual_angles = visual_y * 180
    for i in range(len(visual_likelihoods["head"])):
        if visual_likelihoods["head"][i] < 0.6 or visual_likelihoods["tail"][i] < 0.6:
            predicted_angles[i] = None
            actual_angles[i] = None

    
    np.savetxt("predicted_angles.csv", predicted_angles, delimiter=",", fmt="%s")
    np.savetxt("actual_angles.csv", actual_angles, delimiter=",", fmt="%s")
    return rmse, r2

def main():
    # time_windows = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    time_windows = [200]
    rmses, r2s = [], []
    for time_window in time_windows:
        # TODO optimize function so that it doesn't have to be recalulate same data every time
        rmse, r2 = run_model(time_window)
        rmses.append(rmse)
        r2s.append(r2)
    # plot the RMSE and R-squared values for different time windows
    plt.figure()
    plt.plot(time_windows, rmses, label='RMSE')
    # plt.plot(time_windows, r2s, label='R-squared')
    plt.xlabel('Time Window (ms)')
    plt.ylabel('RMSE')
    # plt.legend()
    plt.title('RMSE for Different Time Windows')
    plt.show()

if __name__ == "__main__":
    main()