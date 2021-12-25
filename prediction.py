import os

import cv2
import numpy as np

import tensorflow as tf


class L1Distance(tf.keras.layers.Layer):
	def __init__(self, *args, **kwargs):
		super().__init__()

	def call(self, encoded_input_image, encoded_validation_image):
		return tf.math.abs(encoded_input_image - encoded_validation_image)


def collect_input_image():
	video = cv2.VideoCapture(0)
	face_cascade = cv2.CascadeClassifier('face_cascade.xml')

	while True:
		_, bgr_image = video.read()
		rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
		gray_frame = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

		cv2.imshow("Capturing", bgr_image)

		faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors = 5)

		for x, y, w, h in faces:
			pass

		key = cv2.waitKey(1)
		if key == 27: break

	video.release()
	cv2.destroyAllWindows()


def predict():
	pass


def normalize_tensor_data(image):
	rgb_tensor = tf.convert_to_tensor(image, dtype = tf.float32)
	rgb_tensor = tf.expand_dims(rgb_tensor, 0)

	rgb_tensor = rgb_tensor / 255

	return rgb_tensor


def normalize_file_data(file):
	byte_image = tf.io.read_file(file)
	image = tf.io.decode_jpeg(byte_image)

	image = image / 255

	return image


CURRENT_DIRECTORY = os.getcwd()
ANCHOR_IMAGE_DIRECTORY = "Validation"
ANCHOR_IMAGE_PATH = os.path.join(os.path.abspath(CURRENT_DIRECTORY), ANCHOR_IMAGE_DIRECTORY)

ANCHOR_IMAGE_DATASET = tf.data.Dataset.list_files(ANCHOR_IMAGE_PATH)

# model = tf.keras.models.load_model('face.h5', 
# 										custom_objects = {'L1Distance': L1Distance})

validate_image = ANCHOR_IMAGE_DATASET.map(normalize_file_data)

data = validate_image.as_numpy_iterator().next()
print(data.shape)