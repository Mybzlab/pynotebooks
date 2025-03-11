import deprecated.analyze_videos as av
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential

def train_classifier(X_train, y_train, epochs=100, batch_size=32):
    """
    Train a neural network with 6 inputs, 3 outputs, and 2 fully connected hidden layers.
    
    Parameters:
    X_train (numpy.ndarray): Training data with shape (num_samples, 6)
    y_train (numpy.ndarray): Training labels with shape (num_samples, 3)
    epochs (int): Number of epochs to train the model
    batch_size (int): Batch size for training
    
    Returns:
    model (tf.keras.Model): Trained neural network model
    """
    # Create a Sequential model
    model = Sequential()
    
    # Add input layer with 6 inputs
    model.add(Dense(64, input_dim=6, activation='relu'))
    
    # Add first fully connected hidden layer
    model.add(Dense(64, activation='relu'))
    
    # Add second fully connected hidden layer
    model.add(Dense(64, activation='relu'))
    
    # Add output layer with 3 outputs
    model.add(Dense(3, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    return model



video_path = r'A:/Uni hdd/Thesis/DLC/Videos/locust/locust.mp4'
csv_path = r'A:/Uni hdd/Thesis/DLC/Videos/locust/model 2/locustDLC_resnet50_Locust2Sep13shuffle1_100000.csv'

brightness_values, directions, angles = av.analyze_data(video_path, csv_path)
print(len(brightness_values["head"]))
# extract features from brightness_values and directions
X = np.array([brightness_values["frontright"], brightness_values["frontleft"], brightness_values["midright"], brightness_values["midleft"], brightness_values["backright"], brightness_values["backleft"]])

# flip axes to have shape (num_samples, 6)
X = np.transpose(X)

# print(X_train)
# print(X_train.shape)

for i in range(len(directions)):
    if directions[i] == "straight":
        directions[i] = [0, 1, 0]
    elif directions[i] == "left":
        directions[i] = [1, 0, 0]
    else:
        directions[i] = [0, 0, 1]
y = directions
# print(y_train)
y = np.array(y)

# create train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Example usage (assuming X_train and y_train are already defined)
model = train_classifier(X_train, y_train)

# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Loss: {loss}, Accuracy: {accuracy}')

predicted_directions = []
for prediction in model.predict(X):
    if np.argmax(prediction) == 0:
        predicted_directions.append("left")
    elif np.argmax(prediction) == 1:
        predicted_directions.append("straight")
    else:
        predicted_directions.append("right")

np.savetxt("predicted_directions.csv", predicted_directions, delimiter=",", fmt="%s")