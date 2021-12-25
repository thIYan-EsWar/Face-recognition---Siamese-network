import os
import uuid

import cv2
import numpy as np

from keras.preprocessing.image import ImageDataGenerator  


class Logger(object):
	@staticmethod
	def log_warning(message: str = ""):
		print("⚠️" + message)

	@staticmethod
	def log_output(message: str = ""):
		print("->" + message)


def collect_data_from_directory(from_path, to_path, show_extraction = False):

	# To extract face data using HAAR feature extraction methods
    face_cascade = cv2.CascadeClassifier('face_cascade.xml')

    # To save the cropped image to a directory
    if os.path.exists(from_path):

    	# Iterating through all the images in directory to extract relevant
    	# data
        for file in os.listdir(from_path):

        	# Opening image file
            image_path = os.path.join(from_path, file)
            image = cv2.imread(image_path, 1)
            
            # Extracting frontal-face data from the image
            faces = face_cascade.detectMultiScale(image, 1.2, 4)

            if len(faces) > 0:
                x, y, w, h = faces[0]

                # Image cropping
                image_extract = image[y: y + h, x: x + w]

                # Logging the message
                Logger.log_output(f'Extracting and saving {file}')
                
                # Saving the cropped image data to drive
                cv2.imwrite(os.path.join(to_path, file), cv2.resize(image_extract, (105, 105)))
                
                # Shows the visual output of the extraction
                if show_extraction:
                	cv2.imshow('Extracting', image_extract)
                	key = cv2.waitKey(20)
                	if key == 27: break

    cv2.destroyAllWindows()
    return


def collect_data_from_camera(to_path, show_image = False):

	SAVE_FILE = ord('s')

	# Creating a camera stream instance
	video = cv2.VideoCapture(0)

	# To extract face data using HAAR feature extraction methods
	face_cascade = cv2.CascadeClassifier("face_cascade.xml")

	while video.isOpened():
		_, frame = video.read(0)
		
		key = cv2.waitKey(1)
		if key == 27: break

		# Converting three channel image into a single channel image
		mono_chrome = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		if frame.shape[0] == 0:
			# Logging the warning if OpenCV can not access the camera
			Logger.log_warning("On-board camera not detected!") 
			break
		
		# Extracting frontal-face data from the image
		faces = face_cascade.detectMultiScale(mono_chrome, 1.2, 5)

		if len(faces) > 0:
			x, y, w, h = faces[0]
			image_path = os.path.join(to_path, f'{uuid.uuid1()}.jpg')
			if key == SAVE_FILE:
				cv2.imwrite(image_path, cv2.resize(frame[y: y + h, x: x + w], (105, 105)))

			# To draw bounding boxes around the face
			if show_image:
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 3)

		# Shows the visual output of the extraction
		if show_image:
			cv2.imshow("Image", frame)

	video.release()
	cv2.destroyAllWindows()


def data_generator(directory):

	# Creating new data for positive label
	datagen = ImageDataGenerator(
		rotation_range = 45,
		width_shift_range = 0.2,
		height_shift_range = 0.2,
		shear_range = 0.2,
		zoom_range = 0.2,
		horizontal_flip = True,
		vertical_flip = True,
		fill_mode = "reflect")

	count = 0
	for batch in datagen.flow_from_directory(
		directory = directory,
		batch_size = 16,
		target_size = (105, 105),
		color_mode = "rgb",
		save_to_dir = "Augumented",
		save_format = "jpg"):

		if count >= 15: break
		count += 1

	return


current_directory = os.path.abspath(path=os.getcwd())

'''
To extract data from negative data
'''
# collect_data_from_directory(
#     from_path=os.path.join(current_directory, "negatives_unprocessed"),
#     to_path=os.path.join(current_directory, "Negatives"))
'''
To collect positive data for the training 
'''
# Label1 = r"Classes\Thiyaneswar"
# collect_data_from_camera(to_path=os.path.join(current_directory, Label1), show_image = True)

'''
To generate positive, augumented data for training
'''
# directory = os.path.join(current_directory, "Classes/")
# data_generator(directory = directory)

'''
To generate anchor image data
'''
# collect_data_from_camera(to_path = os.path.join(current_directory, r"Anchor\Thiyaneswar"), show_image = True)
# directory = os.path.join(current_directory, "Anchor/")
# data_generator(directory = directory)

