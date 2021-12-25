import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPool2D, Input, Flatten 
import tensorflow as tf

import cv2
import numpy as np


def model_encoding():
	input_layer = Input(shape = (105, 105, 1), name = 'input_image')

	conv_layer_1 = Conv2D(64, (10, 10), activation = 'relu')(input_layer)
	max_pooling_1 = MaxPool2D(64, (2, 2), padding = 'same')(conv_layer_1) 

	conv_layer_2 = Conv2D(128, (7, 7), activation = 'relu')(max_pooling_1)
	max_pooling_2 = MaxPool2D(64, (2, 2), padding = 'same')(conv_layer_2) 

	conv_layer_3 = Conv2D(128, (4, 4), activation = 'relu')(max_pooling_2)
	max_pooling_3 = MaxPool2D(64, (2, 2), padding = 'same')(conv_layer_3) 

	conv_layer_4 = Conv2D(256, (4, 4), activation = 'relu')(input_layer)
	flatten_layer = Flatten()(conv_layer_4)
	dense_layer = Dense(4096, activation = 'sigmoid')(flatten_layer)

	return Model(inputs = [input_layer], outputs = [dense_layer], name = 'encoding') 


def normalize_data(file_path):
	'''
	To reduce the domain of each features in the image
	from 0->255(np.uint8) to 0->1(np.float)
	'''
	# byte image
	byte_image = tf.io.read_file(file_path)

	# numpy image
	image = tf.io.decode_jpeg(byte_image)

	# Feature normalize data
	image = image / 255

	return image


# To limit GPU growth to prevent data overflow
GPUS = tf.config.experimental.list_physical_devices('GPU')
for GPU in GPUS:
	tf.config.experimental.set_memory_growth(GPU, True)

# File path definition
# Note: For multiclass classification new variables
# for respective file path must be created
CURRENT_DIRECTORY_ABS_PATH = os.path.abspath(os.getcwd())
LABEL1 = r"Classes\Thiyaneswar"
LABEL1_PATH = os.path.join(CURRENT_DIRECTORY_ABS_PATH, LABEL1)
ANCHOR1_PATH = os.path.join(CURRENT_DIRECTORY_ABS_PATH, r"Anchor\Thiyaneswar")
NEGATIVE_PATH = os.path.join(CURRENT_DIRECTORY_ABS_PATH, "Negatives")

# Load dataset from directory
anchor1 = tf.data.Dataset.list_files(ANCHOR1_PATH + r"\*.jpg").take(450)
positive1 = tf.data.Dataset.list_files(LABEL1_PATH + r"\*.jpg").take(450)
negative = tf.data.Dataset.list_files(NEGATIVE_PATH + r"\*.jpg").take(450)

# Labelling data
positive_labels = tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor1)))
negative_labels = tf.data.Dataset.from_tensor_slices(tf.zeros(len(negative)))

positives = tf.data.Dataset.zip((anchor1, positive1, positive_labels))
negatives = tf.data.Dataset.zip((anchor1, negative, negative_labels))
data_set = positives.concatenate(negatives)

# Preprocess data
preprocess_data = lambda image_input, image_validate, label: (normalize_data(image_input), normalize_data(image_validate), label)
data_set = data_set.map(preprocess_data)
data_set = data_set.cache()
data_set = data_set.shuffle(buffer_size = 2048)

# Train data
TRAINING_PERCENTAGE = 0.8
TRAINING_SAMPLE_SIZE = int(len(data_set) * TRAINING_PERCENTAGE)
train_data = data_set.take(TRAINING_SAMPLE_SIZE)
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Test data
TESTING_PERCENTAGE = 0.2
test_data = data_set.skip(TRAINING_SAMPLE_SIZE)
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# Encoding
encoding = model_encoding()
encoding.summary()
