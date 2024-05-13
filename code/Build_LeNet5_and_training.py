import numpy as np
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0,EfficientNetB3
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
import cv2
import os
print(os.listdir("./data/MNISTcsv))
file = open('./data/MNISTcsv/mnist_train.csv')
data_train = pd.read_csv(file)
y_train = np.array(data_train.iloc[:, 0])
x_train = np.array(data_train.iloc[:, 1:])
file = open("./data/MNISTcsv/mnist_test.csv")
data_test = pd.read_csv(file)
y_test = np.array(data_test.iloc[:, 0])
x_test = np.array(data_test.iloc[:, 1:])
n_features_train = x_train.shape[1]
n_samples_train = x_train.shape[0]
n_features_test = x_test.shape[1]
n_samples_test = x_test.shape[0]
print(n_features_train, n_samples_train, n_features_test, n_samples_test)
print(x_train.shape, y_train.shape, x_test.shape)
size_img = 28
threshold_color = 100 / 255
def greyscale(images):
    return np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images])
# Normilize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
X_train1 = np.full((60000, 32, 32, 3), 0.0)
for i, s in enumerate(x_train):
    img = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
    X_train1[i] = img
test3 = np.full((10000, 32, 32, 3), 0.0)
for i, s in enumerate(x_test):
    img = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
    test3[i] = img

Y_train = to_categorical(y_train, 10)
accuracys = []
for n_oneNumber in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]:
    model = EfficientNetB3(weights='imagenet', input_shape=(32, 32, 3), include_top=False)
    model = Sequential()
    # Layer 1: Convolutional Layer
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 3)))
    # Layer 2: Average Pooling Layer
    model.add(AveragePooling2D(pool_size=(2, 2)))
    # Layer 3: Convolutional Layer
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
    # Layer 4: Average Pooling Layer
    model.add(AveragePooling2D(pool_size=(2, 2)))
    # Flatten the output for the fully connected layers
    model.add(Flatten())
    # Layer 5: Fully Connected Layer
    model.add(Dense(units=120, activation='relu'))
    # Layer 6: Fully Connected Layer
    model.add(Dense(units=84, activation='relu'))
    # Output Layer
    model.add(Dense(units=10, activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Display the model summary
    model.summary()
    selected_indices = []
    for label in range(10):
        label_indices = np.where(Y_train[:, label] == 1)[0]
        selected_indices.extend(label_indices[:n_oneNumber])
    X_train_selected = X_train1[selected_indices]
    Y_train_selected = Y_train[selected_indices]
    history = model.fit(X_train_selected, Y_train_selected,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.1,
                        shuffle=True,
                        verbose=2)
    test_predictions = model.predict(test3)
    x_test_resized = np.array([cv2.resize(img, (32, 32)) for img in x_test])
    x_test_resized = x_test_resized.reshape(-1, 32, 32, 1)
    x_test_resized_rgb = np.concatenate([x_test_resized] * 3, axis=-1)
    y_test_one_hot = to_categorical(y_test, num_classes=10)
    accuracy = model.evaluate(x_test_resized_rgb, y_test_one_hot)[1]
    accuracys.append(accuracy)
    print('Accuracy of the model:', accuracy * 100)