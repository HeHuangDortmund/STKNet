import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, GlobalAveragePooling2D, Flatten, \
    AveragePooling2D, Dense
import tensorflow as tf
import keras.backend as K
import os
import math
from scipy.interpolate import interp1d


from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


desired_length = 25
margin = 0.3
noise_level_small = 1 / 3

import random
from itertools import combinations
import pandas as p
from tqdm import tqdm


# Functions for data augmentation
def read_trajectory_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    trajectory = []
    current_segment = []
    for line in lines:
        data = line.strip().split()
        latitude, longitude, occupancy, time = map(float, data)

        if occupancy == 1:
            current_segment.append([latitude, longitude])
        elif current_segment:
            trajectory.append(np.array(current_segment))
            current_segment = []

    if current_segment:
        trajectory.append(np.array(current_segment))

    return trajectory

def read_all_trajectories(folder_path):
    all_trajectories = []
    for filename in tqdm(os.listdir(folder_path), desc="Processing files", unit="file"):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            trajectory = read_trajectory_from_file(file_path)
            all_trajectories.extend(trajectory)
    return all_trajectories

def shift_trajectory(trajectory):
    shift_vector = np.random.uniform(-3, 3, size=2)
    shifted_trajectory = trajectory + shift_vector
    return np.array(shifted_trajectory)

def rotate_trajectory(trajectory):
    pivot_point = trajectory[-1]
    rotation_angle = np.random.uniform(5, 15)
    rotation_angle_rad = math.radians(rotation_angle)
    rotation_matrix = np.array([[np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad)],
                                [np.sin(rotation_angle_rad), np.cos(rotation_angle_rad)]])
    rotated_trajectory = [(rotation_matrix @ (point - pivot_point)) + pivot_point for point in trajectory]
    return np.array(rotated_trajectory)

def remove_random_points(trajectory):
    num_points = trajectory.shape[0]
    remove_percentage = np.random.uniform(0.05, 0.20)
    remove_count = int(num_points * remove_percentage)
    if remove_count > 0:
        if np.random.rand() < 0.5:
            trajectory = trajectory[remove_count:, :]
        else:
            trajectory = trajectory[:-remove_count, :]
    return np.array(trajectory)

def stretch_trajectory(trajectory):
    
    start_idx = random.randint(0, len(trajectory) - 2)  
    end_idx = random.randint(start_idx + 1, len(trajectory) - 1)


    start_point = trajectory[start_idx]
    end_point = trajectory[end_idx]


    factor = random.uniform(0.2, 1.8)

    scaling_matrix = np.array([[factor, 0],
                               [0, factor]])


    scaled_start_point = scaling_matrix @ (start_point - np.mean([start_point, end_point], axis=0)) + np.mean(
        [start_point, end_point], axis=0)
    scaled_end_point = scaling_matrix @ (end_point - np.mean([start_point, end_point], axis=0)) + np.mean(
        [start_point, end_point], axis=0)

 
    stretched_trajectory = []
    for i, point in enumerate(trajectory):
        if start_idx <= i <= end_idx:
  
            scaled_point = scaling_matrix @ (point - np.mean([start_point, end_point], axis=0)) + np.mean(
                [start_point, end_point], axis=0)
            stretched_trajectory.append(scaled_point)
        else:
            stretched_trajectory.append(point)

    return np.array(stretched_trajectory)


# Read the train labels
labels = pd.read_csv("./data/sequences/trainlabels.txt", header=None)
labels_test = pd.read_csv("./data/sequences/testlabels.txt", header=None)
labels = labels.iloc[:, 0].tolist()
labels_test = labels_test.iloc[:, 0].tolist()
# List files in the directory
file_names_points = ["trainimg-" + str(x) + "-points.txt" for x in range(60000)]
file_names_points_test = ["testimg-" + str(x) + "-points.txt" for x in range(10000)]


# Create training data
def createData(bb, n_oneNumber_train):
    indices_of_0 = [index for index, value in enumerate(labels) if value == 0][
                   (bb * n_oneNumber_train):(((bb + 1) * n_oneNumber_train))]
    indices_of_1 = [index for index, value in enumerate(labels) if value == 1][
                   (bb * n_oneNumber_train):(((bb + 1) * n_oneNumber_train))]
    indices_of_2 = [index for index, value in enumerate(labels) if value == 2][
                   (bb * n_oneNumber_train):(((bb + 1) * n_oneNumber_train))]
    indices_of_3 = [index for index, value in enumerate(labels) if value == 3][
                   (bb * n_oneNumber_train):(((bb + 1) * n_oneNumber_train))]
    indices_of_4 = [index for index, value in enumerate(labels) if value == 4][
                   (bb * n_oneNumber_train):(((bb + 1) * n_oneNumber_train))]
    indices_of_5 = [index for index, value in enumerate(labels) if value == 5][
                   (bb * n_oneNumber_train):(((bb + 1) * n_oneNumber_train))]
    indices_of_6 = [index for index, value in enumerate(labels) if value == 6][
                   (bb * n_oneNumber_train):(((bb + 1) * n_oneNumber_train))]
    indices_of_7 = [index for index, value in enumerate(labels) if value == 7][
                   (bb * n_oneNumber_train):(((bb + 1) * n_oneNumber_train))]
    indices_of_8 = [index for index, value in enumerate(labels) if value == 8][
                   (bb * n_oneNumber_train):(((bb + 1) * n_oneNumber_train))]
    indices_of_9 = [index for index, value in enumerate(labels) if value == 9][
                   (bb * n_oneNumber_train):(((bb + 1) * n_oneNumber_train))]

    ind0123456789 = indices_of_0 + indices_of_1 + indices_of_2 + indices_of_3 + indices_of_4 + indices_of_5 + indices_of_6 + indices_of_7 + indices_of_8 + indices_of_9
    # ind0123456789 = indices_of_0 + indices_of_1 + indices_of_2 + indices_of_3 + indices_of_4
    # ind0123456789 = indices_of_0 + indices_of_1 + indices_of_2 + indices_of_3 + indices_of_6 + indices_of_7 + indices_of_8 + indices_of_9
    file_paths = ["./data/sequences/" + file_name for file_name in file_names_points]

    sequence_list = []

    for i in ind0123456789:
        one_sequence = pd.read_csv(file_paths[i])
        one_sequence = one_sequence[(one_sequence >= 0).all(axis=1)]
        one_sequence = np.array(one_sequence)
        sequence_list.append(one_sequence)

    downsampled_sequence = []
    for sequence in sequence_list:
        num_points = sequence.shape[0]
        # Bicubic interpolation for both upsampling and downsampling
        old_indices = np.arange(num_points)
        if num_points >= desired_length:
            new_indices = np.arange(desired_length) * (num_points - 1) / (desired_length - 1)
        else:
            new_indices = np.linspace(0, num_points - 1, desired_length)
        # Bicubic interpolation for x and y coordinates separately
        interp_x = interp1d(old_indices, sequence[:, 0], kind='cubic')
        interp_y = interp1d(old_indices, sequence[:, 1], kind='cubic')
        interpolated_sequence_x = interp_x(new_indices)
        interpolated_sequence_y = interp_y(new_indices)
        sequence1 = np.column_stack((interpolated_sequence_x, interpolated_sequence_y))
        downsampled_sequence.append(sequence1)
    downsampled_sequence = np.array(downsampled_sequence)

    selected_trajectories_subset_shifted = [shift_trajectory(trajectory) for trajectory in sequence_list]
    selected_trajectories_subset_cropped = [remove_random_points(trajectory) for trajectory in sequence_list]
    selected_trajectories_subset_rotated = [rotate_trajectory(trajectory) for trajectory in sequence_list]
    selected_trajectories_subset_stretch = [stretch_trajectory(trajectory) for trajectory in sequence_list]


    downsampled_sequence_shifted = []
    for sequence in selected_trajectories_subset_shifted:
        num_points = sequence.shape[0]
        # Bicubic interpolation for both upsampling and downsampling
        old_indices = np.arange(num_points)
        if num_points >= desired_length:
            new_indices = np.arange(desired_length) * (num_points - 1) / (desired_length - 1)
        else:
            new_indices = np.linspace(0, num_points - 1, desired_length)
        # Bicubic interpolation for x and y coordinates separately
        interp_x = interp1d(old_indices, sequence[:, 0], kind='cubic')
        interp_y = interp1d(old_indices, sequence[:, 1], kind='cubic')
        interpolated_sequence_x = interp_x(new_indices)
        interpolated_sequence_y = interp_y(new_indices)
        sequence1 = np.column_stack((interpolated_sequence_x, interpolated_sequence_y))
        downsampled_sequence_shifted.append(sequence1)

    downsampled_sequence_cropped = []
    for sequence in selected_trajectories_subset_cropped:
        num_points = sequence.shape[0]
        # Bicubic interpolation for both upsampling and downsampling
        old_indices = np.arange(num_points)
        if num_points >= desired_length:
            new_indices = np.arange(desired_length) * (num_points - 1) / (desired_length - 1)
        else:
            new_indices = np.linspace(0, num_points - 1, desired_length)
        # Bicubic interpolation for x and y coordinates separately
        interp_x = interp1d(old_indices, sequence[:, 0], kind='cubic')
        interp_y = interp1d(old_indices, sequence[:, 1], kind='cubic')
        interpolated_sequence_x = interp_x(new_indices)
        interpolated_sequence_y = interp_y(new_indices)
        sequence1 = np.column_stack((interpolated_sequence_x, interpolated_sequence_y))
        downsampled_sequence_cropped.append(sequence1)

    downsampled_sequence_rotated = []
    for sequence in selected_trajectories_subset_rotated:
        num_points = sequence.shape[0]
        # Bicubic interpolation for both upsampling and downsampling
        old_indices = np.arange(num_points)
        if num_points >= desired_length:
            new_indices = np.arange(desired_length) * (num_points - 1) / (desired_length - 1)
        else:
            new_indices = np.linspace(0, num_points - 1, desired_length)
        # Bicubic interpolation for x and y coordinates separately
        interp_x = interp1d(old_indices, sequence[:, 0], kind='cubic')
        interp_y = interp1d(old_indices, sequence[:, 1], kind='cubic')
        interpolated_sequence_x = interp_x(new_indices)
        interpolated_sequence_y = interp_y(new_indices)
        sequence1 = np.column_stack((interpolated_sequence_x, interpolated_sequence_y))
        downsampled_sequence_rotated.append(sequence1)

    downsampled_sequence_stretch = []
    for sequence in selected_trajectories_subset_stretch:
        num_points = sequence.shape[0]
        # Bicubic interpolation for both upsampling and downsampling
        old_indices = np.arange(num_points)
        if num_points >= desired_length:
            new_indices = np.arange(desired_length) * (num_points - 1) / (desired_length - 1)
        else:
            new_indices = np.linspace(0, num_points - 1, desired_length)
        # Bicubic interpolation for x and y coordinates separately
        interp_x = interp1d(old_indices, sequence[:, 0], kind='cubic')
        interp_y = interp1d(old_indices, sequence[:, 1], kind='cubic')
        interpolated_sequence_x = interp_x(new_indices)
        interpolated_sequence_y = interp_y(new_indices)
        sequence1 = np.column_stack((interpolated_sequence_x, interpolated_sequence_y))
        downsampled_sequence_stretch.append(sequence1)

    downsampled_sequence_shifted = np.array(downsampled_sequence_shifted)
    downsampled_sequence_rotated = np.array(downsampled_sequence_rotated)
    downsampled_sequence_cropped = np.array(downsampled_sequence_cropped)
    downsampled_sequence_stretch = np.array(downsampled_sequence_stretch)

    downsampled_sequence_shifted_noise = downsampled_sequence_shifted + np.random.normal(0, noise_level_small,
                                                                                         downsampled_sequence_shifted.shape)
    downsampled_sequence_rotated_noise = downsampled_sequence_rotated + np.random.normal(0, noise_level_small,
                                                                                         downsampled_sequence_rotated.shape)
    downsampled_sequence_cropped_noise = downsampled_sequence_cropped + np.random.normal(0, noise_level_small,
                                                                                         downsampled_sequence_cropped.shape)
    downsampled_sequence_stretch_noise = downsampled_sequence_stretch + np.random.normal(0, noise_level_small,
                                                                                         downsampled_sequence_stretch.shape)

    combined_data = np.concatenate([downsampled_sequence,
                                    downsampled_sequence_shifted_noise,
                                    downsampled_sequence_rotated_noise,
                                    downsampled_sequence_cropped_noise,
                                    downsampled_sequence_stretch_noise], axis=0)

    labels2 = np.repeat(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), n_oneNumber_train)  # Original labels
    combined__labels = np.tile(labels2, 5)
    n_tra = len(combined_data)
    # Generate all combinations of three elements from the range 0 to 199
    combinations_list = np.array(list(combinations(range(n_tra), 2)))
    x1 = np.array([combined_data[i] for i in combinations_list[:, 0]])
    x2 = np.array([combined_data[i] for i in combinations_list[:, 1]])
    train_x = [x1, x2]
    y1 = np.array([combined__labels[i] for i in combinations_list[:, 0]])
    y2 = np.array([combined__labels[i] for i in combinations_list[:, 1]])
    train_y = np.where(y1 == y2, 0, 1).astype(np.float32)
    return train_x, train_y, downsampled_sequence, combined__labels, combined_data


class SpatialTemporalConvolution(tf.keras.layers.Layer):
    def __init__(self, spatial_kernel_size, spatial_filters, **kwargs):
        super(SpatialTemporalConvolution, self).__init__(**kwargs)
        self.spatial_kernel_size = spatial_kernel_size
        self.spatial_filters = spatial_filters

    def build(self, input_shape):
        self.spatial_conv = tf.keras.layers.Conv2D(self.spatial_filters, self.spatial_kernel_size, padding='same')
        self.temporal_kernel = tf.constant_initializer(1.0)  

    def call(self, spatial_input):
        batch_size = tf.shape(spatial_input)[0]
        spatial_features = self.spatial_conv(spatial_input)

        # time distance matrix
        seq = tf.cast(tf.range(1, tf.shape(spatial_input)[1] + 1), tf.float32)
        time_distances = tf.abs(tf.expand_dims(seq, -1) - tf.expand_dims(seq, 0))
        time_distances = tf.expand_dims(tf.expand_dims(time_distances, 0), -1)

        # temporal convolution

        temporal_features = tf.nn.conv2d(time_distances,
                                         tf.ones((self.spatial_kernel_size[0], self.spatial_kernel_size[1], 1, 1)) * (
                                                     1 / (self.spatial_kernel_size[0] * self.spatial_kernel_size[1])),
                                         strides=(1, 1, 1, 1), padding='SAME')
        """
        temporal_features = tf.nn.conv2d(time_distances,
                                         tf.ones((self.spatial_kernel_size[0], self.spatial_kernel_size[1], 1, 1)),
                                         strides=(1, 1, 1, 1), padding='SAME')
        """

        return spatial_features * temporal_features
class DistanceMatrixLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        traj1, traj2 = inputs
        traj1 = tf.expand_dims(traj1, axis=2)
        traj2 = tf.expand_dims(traj2, axis=1)
        distances = tf.reduce_sum(tf.square(traj1 - traj2), axis=-1)
        return distances

class GetGlobalForm(Layer):
    def __init__(self, **kwargs):
        super(GetGlobalForm, self).__init__(**kwargs)

    def call(self, inputs):
    
        matrices = []
        for size in range(5, 13):
            n1 = inputs.shape[1]
            indices1 = [i * (n1 - 1) // (size - 1) for i in range(size)]
            sub_matrix = tf.gather(inputs, indices1, axis=1)
            sub_matrix = tf.gather(sub_matrix, indices1, axis=2)
        
            pad_rows = 12 - sub_matrix.shape[1]
            pad_cols = 12 - sub_matrix.shape[2]
    
            sub_matrix_padded = tf.pad(sub_matrix, paddings=[(0, 0), (0, pad_rows), (0, pad_cols)])
            matrices.append(sub_matrix_padded)
        resultts = tf.stack(matrices, axis=-1)  # Specify axis=3
        return resultts

class GlobalMaxPooling2D(Layer):
    def __init__(self, **kwargs):
        super(GlobalMaxPooling2D, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_max(inputs, axis=[1, 2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class GlobalInfoProcess(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalInfoProcess, self).__init__(**kwargs)


        self.conv1 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(6, 5))
        self.conv2 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(6, 6))
        self.conv3 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(6, 7))
        self.conv4 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(7, 6))
        self.conv5 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(7, 7))
        self.conv6 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(7, 8))
        self.conv7 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(8, 7))
        self.conv8 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(8, 8))
        self.conv9 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(8, 9))
        self.conv10 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(9, 8))
        self.conv11 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(9, 9))



        self.batch_norm = BatchNormalization()

        self.global_max_pooling = GlobalMaxPooling2D()
        self.global_avg_pooling = AveragePooling2D(pool_size=(3, 3))

        self.flatten = Flatten()

    def call(self, inputs):
        # BatchNormalization
        x = self.batch_norm(inputs)

        # Convolutional layers
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        conv3_out = self.conv3(x)
        conv4_out = self.conv4(x)
        conv5_out = self.conv5(x)
        conv6_out = self.conv6(x)
        conv7_out = self.conv7(x)
        conv8_out = self.conv8(x)
        conv9_out = self.conv9(x)
        conv10_out = self.conv10(x)
        conv11_out = self.conv11(x)



        # Pooling layers

        concatenate1 = self.flatten(tf.concat(
            [self.global_avg_pooling(conv1_out),
                self.global_avg_pooling(conv2_out),
                self.global_avg_pooling(conv3_out),
                self.global_avg_pooling(conv4_out),
                self.global_avg_pooling(conv5_out),
                self.global_avg_pooling(conv6_out),
                self.global_avg_pooling(conv7_out),
                self.global_avg_pooling(conv8_out),
                self.global_avg_pooling(conv9_out),
                self.global_avg_pooling(conv10_out),
                self.global_avg_pooling(conv11_out)
            ],
            axis=-1))

        concatenate2 = self.flatten(tf.concat(
            [ self.global_max_pooling(conv1_out),
                self.global_max_pooling(conv2_out),
                self.global_max_pooling(conv3_out),
                self.global_max_pooling(conv4_out),
                self.global_max_pooling(conv5_out),
                self.global_max_pooling(conv6_out),
                self.global_max_pooling(conv7_out),
                self.global_max_pooling(conv8_out),
                self.global_max_pooling(conv9_out),
                self.global_max_pooling(conv10_out),
                self.global_max_pooling(conv11_out)
            ],
            axis=-1))

        concatenated = tf.concat([concatenate1, concatenate2], axis=-1)

        return concatenated

class LocalInfoProcess(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LocalInfoProcess, self).__init__(**kwargs)
        self.weight_decay = 0.001  # 'default'
        # ... (your layer definitions here)
        self.batch_norm = BatchNormalization()

        self.conv33 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(3, 3))
        self.conv34 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(3, 4))
        self.conv35 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(3, 5))
        self.conv43 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(4, 3))
        self.conv44 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(4, 4))
        self.conv45 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(4, 5))
        self.conv46 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(4, 6))
        self.conv53 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(5, 3))
        self.conv54 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(5, 4))
        self.conv55 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(5, 5))
        self.conv56 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(5, 6))
        self.conv64 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(6, 4))
        self.conv65 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(6, 5))
        self.conv66 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(6, 6))

        self.concat = keras.layers.Concatenate()
        self.flatten = Flatten()

        self.avg_pool33 = AveragePooling2D(pool_size=(3, 3))
        self.global_max_pooling = GlobalMaxPooling2D()

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, axis=-1)
        inputs = self.batch_norm(inputs)

        conv33 = self.conv33(inputs)
        conv34 = self.conv34(inputs)
        conv35 = self.conv35(inputs)
        conv43 = self.conv43(inputs)
        conv44 = self.conv44(inputs)
        conv45 = self.conv45(inputs)
        conv46 = self.conv46(inputs)
        conv53 = self.conv53(inputs)
        conv54 = self.conv54(inputs)
        conv55 = self.conv55(inputs)
        conv56 = self.conv56(inputs)
        conv64 = self.conv64(inputs)
        conv65 = self.conv65(inputs)
        conv66 = self.conv66(inputs)

        # Apply AveragePooling2D to each convolutional layer
        avg_pool33 = self.avg_pool33(conv33)
        avg_pool34 = self.avg_pool33(conv34)
        avg_pool35 = self.avg_pool33(conv35)
        avg_pool43 = self.avg_pool33(conv43)
        avg_pool44 = self.avg_pool33(conv44)
        avg_pool45 = self.avg_pool33(conv45)
        avg_pool46 = self.avg_pool33(conv46)
        avg_pool53 = self.avg_pool33(conv53)
        avg_pool54 = self.avg_pool33(conv54)
        avg_pool55 = self.avg_pool33(conv55)
        avg_pool56 = self.avg_pool33(conv56)
        avg_pool64 = self.avg_pool33(conv64)
        avg_pool65 = self.avg_pool33(conv65)
        avg_pool66 = self.avg_pool33(conv66)

        flat_avg_pool33 = keras.layers.Flatten()(avg_pool33)
        flat_avg_pool34 = keras.layers.Flatten()(avg_pool34)
        flat_avg_pool35 = keras.layers.Flatten()(avg_pool35)
        flat_avg_pool43 = keras.layers.Flatten()(avg_pool43)
        flat_avg_pool44 = keras.layers.Flatten()(avg_pool44)
        flat_avg_pool45 = keras.layers.Flatten()(avg_pool45)
        flat_avg_pool46 = keras.layers.Flatten()(avg_pool46)
        flat_avg_pool53 = keras.layers.Flatten()(avg_pool53)
        flat_avg_pool54 = keras.layers.Flatten()(avg_pool54)
        flat_avg_pool55 = keras.layers.Flatten()(avg_pool55)
        flat_avg_pool56 = keras.layers.Flatten()(avg_pool56)
        flat_avg_pool64 = keras.layers.Flatten()(avg_pool64)
        flat_avg_pool65 = keras.layers.Flatten()(avg_pool65)
        flat_avg_pool66 = keras.layers.Flatten()(avg_pool66)

        concatenate1 = self.flatten(tf.concat(
            [self.global_max_pooling(conv33),
             self.global_max_pooling(conv34),
             self.global_max_pooling(conv35),
             self.global_max_pooling(conv43),
             self.global_max_pooling(conv44),
             self.global_max_pooling(conv45),
             self.global_max_pooling(conv46),
             self.global_max_pooling(conv53),
             self.global_max_pooling(conv54),
             self.global_max_pooling(conv55),
             self.global_max_pooling(conv56),
             self.global_max_pooling(conv64),
             self.global_max_pooling(conv65),
             self.global_max_pooling(conv66)],
            axis=-1))

        concatenate2 = self.concat(
            [flat_avg_pool33,
             flat_avg_pool34,
             flat_avg_pool35,
             flat_avg_pool43,
             flat_avg_pool44,
             flat_avg_pool45,
             flat_avg_pool46,
             flat_avg_pool53,
             flat_avg_pool54,
             flat_avg_pool55,
             flat_avg_pool56,
             flat_avg_pool64,
             flat_avg_pool65,
             flat_avg_pool66])

        concatenated = tf.concat([concatenate1, concatenate2], axis=-1)

        return concatenated


class CombineLayer(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(CombineLayer, self).__init__(**kwargs)
        self.concat = keras.layers.Concatenate(axis=-1)
        self.fc1 = keras.layers.Dense(units=64, activation='tanh')
        self.fc2 = keras.layers.Dense(units=16, activation='tanh')
        self.fc3 = keras.layers.Dense(units=4, activation='tanh')
        self.fc4 = keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs):
        concatenated = self.concat(inputs)
        fc1 = self.fc1(concatenated)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        output = self.fc4(fc3)
        return output

def custom_loss(y_true, y_pred):
    lossAN = K.maximum(0.0, margin - y_pred)**2
    lossAP = y_pred**2
    loss = (1-y_true)*lossAP + y_true*lossAN
    return loss


def create_test(digit):
    indices_of = [index for index, value in enumerate(labels_test) if value == digit]
    file_paths_test = ["./data/sequences/" + file_name for file_name in file_names_points_test]
    sequence_list_test = []
    for i in indices_of:
        one_sequence = pd.read_csv(file_paths_test[i])
        one_sequence = one_sequence[(one_sequence >= 0).all(axis=1)]
        one_sequence = np.array(one_sequence)
        sequence_list_test.append(one_sequence)
    downsampled_sequence_test = []
    for sequence in sequence_list_test:
        num_points = sequence.shape[0]
        # Bicubic interpolation for both upsampling and downsampling
        old_indices = np.arange(num_points)
        if num_points >= desired_length:
            new_indices = np.arange(desired_length) * (num_points - 1) / (desired_length - 1)
        else:
            new_indices = np.linspace(0, num_points - 1, desired_length)
        # Bicubic interpolation for x and y coordinates separately
        interp_x = interp1d(old_indices, sequence[:, 0], kind='cubic')
        interp_y = interp1d(old_indices, sequence[:, 1], kind='cubic')
        interpolated_sequence_x = interp_x(new_indices)
        interpolated_sequence_y = interp_y(new_indices)
        sequence1 = np.column_stack((interpolated_sequence_x, interpolated_sequence_y))
        downsampled_sequence_test.append(sequence1)
    return downsampled_sequence_test

def model_predict_test(digit, n_oneNumber_train=20):
    k = math.ceil(math.sqrt(n_oneNumber_train))
    downsampled_sequence_test = create_test(digit)
    n_test = len(downsampled_sequence_test)
    new_list1 = []
    for _ in range(n_test):
        new_list1.extend(downsampled_sequence)
    new_list2 = []
    for array in downsampled_sequence_test:
        for _ in range(n_oneNumber_train * 10):
            new_list2.append(array)
    to_vote_x = [np.array(new_list1), np.array(new_list2)]
    distances = model.predict(to_vote_x)
    distances_2d = np.reshape(distances, (n_test, n_oneNumber_train * 10))

    label_train = np.repeat(np.arange(10), n_oneNumber_train)
    top_k_indices = np.argpartition(distances_2d, k, axis=1)[:, :k]
    def find_mode_along_rows(array):
    
        mode_along_rows = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=array)

        return mode_along_rows
    modes = find_mode_along_rows(label_train[top_k_indices])
    distances_DTW = np.zeros((len(to_vote_x[0])))
    for i in tqdm(range(len(to_vote_x[0])), desc='Computing DTW Distances', total=len(to_vote_x[0])):
        distance_DTW, _ = fastdtw(to_vote_x[0][i], to_vote_x[1][i], dist=euclidean)
        distances_DTW[i] = distance_DTW
    distances_2d_DTW = np.reshape(distances_DTW, (n_test, n_oneNumber_train * 10))
    label_train_DTW = np.repeat(np.arange(10), n_oneNumber_train)
    top_k_indices_DTW = np.argpartition(distances_2d_DTW, k, axis=1)[:, :k]
    def find_mode_along_rows_DTW(array):
       
        mode_along_rows = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=array)
        return mode_along_rows
    modes_DTW = find_mode_along_rows_DTW(label_train_DTW[top_k_indices_DTW])
    return modes, modes_DTW

trajectory_length = desired_length
num_dimensions = 2
sequence_length = desired_length
def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape), 
        Dense(num_classes, activation='softmax') 
    ])

    return model

def create_test_lstm(digit):
    indices_of = [index for index, value in enumerate(labels_test) if value == digit]
    file_paths_test = ["./data/sequences/" + file_name for file_name in file_names_points_test]
    sequence_list_test = []
    for i in indices_of:
        one_sequence = pd.read_csv(file_paths_test[i])
        one_sequence = one_sequence[(one_sequence >= 0).all(axis=1)]
        one_sequence = np.array(one_sequence)
        sequence_list_test.append(one_sequence)
    downsampled_sequence_test = []
    for sequence in sequence_list_test:
        num_points = sequence.shape[0]
        # Bicubic interpolation for both upsampling and downsampling
        old_indices = np.arange(num_points)
        if num_points >= desired_length:
            new_indices = np.arange(desired_length) * (num_points - 1) / (desired_length - 1)
        else:
            new_indices = np.linspace(0, num_points - 1, desired_length)
        # Bicubic interpolation for x and y coordinates separately
        interp_x = interp1d(old_indices, sequence[:, 0], kind='cubic')
        interp_y = interp1d(old_indices, sequence[:, 1], kind='cubic')
        interpolated_sequence_x = interp_x(new_indices)
        interpolated_sequence_y = interp_y(new_indices)
        sequence1 = np.column_stack((interpolated_sequence_x, interpolated_sequence_y))
        downsampled_sequence_test.append(sequence1)
    return np.array(downsampled_sequence_test)

features = 2
num_classes = 10


for n_oneNumber_train in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]:
    print("Current BB:", BB)
    print("Current n_oneNumber_train:", n_oneNumber_train)
    train_x, train_y, downsampled_sequence, combined__labels, combined_data = createData(bb=BB,
                                                                                         n_oneNumber_train=n_oneNumber_train)
    np.random.seed(1000)
    input1 = keras.layers.Input(shape=(trajectory_length, num_dimensions))
    input2 = keras.layers.Input(shape=(trajectory_length, num_dimensions))
    haversine_layer = DistanceMatrixLayer()
    distance1 = haversine_layer([input1, input2])
    get_global_info = GetGlobalForm()
    global_info1 = get_global_info(distance1)
    global_info_process = GlobalInfoProcess()
    local_info_process = LocalInfoProcess()
    global_info_vec1 = global_info_process(global_info1)
    local_info_vec1 = local_info_process(distance1)
    combine_local_global = CombineLayer()
    output1 = combine_local_global([local_info_vec1, global_info_vec1])
    model = keras.models.Model(inputs=[input1, input2], outputs=output1)
    model.compile(loss=custom_loss, optimizer=keras.optimizers.Adam(learning_rate=0.001))  # -0000
    model.fit(train_x, train_y, epochs=30, batch_size=1024, shuffle=True)

    model_lstm = create_lstm_model((sequence_length, features), num_classes)
  
    optimizer = Adam(learning_rate=0.001)
    model_lstm.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_lstm.fit(combined_data, combined__labels, batch_size=1024, epochs=1000)

    for h in range(10):
        print("Current digit:", h)
        predicted_label_test, predicted_label_test_dtw = model_predict_test(digit=h,
                                                                            n_oneNumber_train=n_oneNumber_train)
        true_labels_digit = np.repeat(h, predicted_label_test.shape[0])
        filename_predicted_label_test = './results/Proposed/BB' + str(BB) + '/predicted_label_test_for_digit_' + str(
            h) + '_nTrain_' + str(n_oneNumber_train) + '.npy'
        filename_true_labels_digit = './results/Proposed/BB' + str(BB) + '/true_label_test_for_digit_' + str(
            h) + '_nTrain_' + str(n_oneNumber_train) + '.npy'
        np.save(filename_predicted_label_test, predicted_label_test)
        np.save(filename_true_labels_digit, true_labels_digit)
        accuracy = np.mean(predicted_label_test == true_labels_digit)
        print("AccuracyProposed:", accuracy)
        filename_predicted_label_test_dtw = './results/DTW/BB' + str(BB) + '/predicted_label_test_for_digit_' + str(
            h) + '_nTrain_' + str(n_oneNumber_train) + '.npy'
        filename_true_labels_digit_dtw = './results/DTW/BB' + str(BB) + '/true_label_test_for_digit_' + str(
            h) + '_nTrain_' + str(n_oneNumber_train) + '.npy'
        np.save(filename_predicted_label_test_dtw, predicted_label_test_dtw)
        np.save(filename_true_labels_digit_dtw, true_labels_digit)
        accuracy = np.mean(predicted_label_test_dtw == true_labels_digit)
        print("AccuracyDTW:", accuracy)
        downsampled_sequence_test_lstm = create_test_lstm(h)
        predicted_label_test_lstm = model_lstm.predict(downsampled_sequence_test_lstm)
        predicted_labels_lstm = np.argmax(predicted_label_test_lstm, axis=1)
        true_labels_digit_lstm = np.repeat(h, predicted_label_test_lstm.shape[0])
        filename_predicted_label_test_lstm = './results/LSTM/BB' + str(BB) + '/predicted_label_test_for_digit_' + str(
            h) + '_nTrain_' + str(n_oneNumber_train) + '.npy'
        filename_true_labels_digit_lstm = './results/LSTM/BB' + str(BB) + '/true_label_test_for_digit_' + str(
            h) + '_nTrain_' + str(n_oneNumber_train) + '.npy'
        np.save(filename_predicted_label_test_lstm, predicted_labels_lstm)
        np.save(filename_true_labels_digit_lstm, true_labels_digit_lstm)
        accuracy = np.mean(predicted_labels_lstm == true_labels_digit_lstm)
        print("AccuracyLSTM:", accuracy)

















