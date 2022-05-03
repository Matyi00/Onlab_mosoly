# pkill -KILL -u polyamatyas
# watch -n 0 nvidia-smi
##
import numpy as np
import imutils
import cv2
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import mediapipe as mp
import pandas as pd
import os
import pickle
##
def unit_vector(vector):
	""" Returns the unit vector of the vector.  """
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	""" Returns the angle in radians between vectors 'v1' and 'v2'::

			>>> angle_between((1, 0, 0), (0, 1, 0))
			1.5707963267948966
			>>> angle_between((1, 0, 0), (1, 0, 0))
			0.0
			>>> angle_between((1, 0, 0), (-1, 0, 0))
			3.141592653589793
	"""
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

##

dir_list = os.listdir(r"/home/polyamatyas/projects/mosoly/AMFEDPLUS_Distribution/Landmark Points (labeled videos)")
files = [text.split(".")[0] for text in dir_list if (os.path.getsize(r"/home/polyamatyas/projects/mosoly/AMFEDPLUS_Distribution/Landmark Points (labeled videos)/" + text) != 0)]



##
file = open(r'/home/polyamatyas/projects/mosoly/negative_positive_ratio.pkl', 'rb')
file_negative_positive = pickle.load(file)
file.close()

##

training_rate, validation_rate, test_rate = 0.8, 0.1, 0.1
np.random.shuffle(files)
len_files = len(files)

training_files = files[:int(len_files * training_rate)]
validation_files = files[int(len_files * training_rate):int(len_files * (training_rate + validation_rate))]
test_files = files[int(len_files * (training_rate + validation_rate)):]

number_neg_pos = [0, 0]

for file in training_files:
	number_neg_pos[0] += file_negative_positive[file][0]
	number_neg_pos[1] += file_negative_positive[file][1]

weight_for_0 = (1 / number_neg_pos[0]) * ((number_neg_pos[0] + number_neg_pos[1]) / 2.0)
weight_for_1 = (1 / number_neg_pos[1]) * ((number_neg_pos[0] + number_neg_pos[1]) / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

##
def dataset_generator(augment_number, file_list):
	count = 0
	for file in file_list:
		# file = r"3f28a162-f529-4bc4-bbd6-f1ecf92d8b22"
		# file = r"0005b896-33ad-4ecf-93b4-dad735ad69b6"
		label_csv = pd.read_csv(r"/home/polyamatyas/projects/mosoly/AMFEDPLUS_Distribution/AU Labels/" + file + "-label.csv")
		len_label = label_csv.shape[0]
		label_idx = 0


		landmarks = pd.read_csv(
			r"/home/polyamatyas/projects/mosoly/AMFEDPLUS_Distribution/Landmark Points (labeled videos)/" + file + ".csv")
		cap = cv2.VideoCapture(
			r"/home/polyamatyas/projects/mosoly/AMFEDPLUS_Distribution/Videos - FLV (labeled)/" + file + ".flv")

		frame_number = 0
		while cap.isOpened():
			frame_exists, image = cap.read()
			if not frame_exists:
				break
			try:
				left_eye = (int((np.float64(landmarks.pt_affdex_tracker_34[frame_number]) + np.float64(
					landmarks.pt_affdex_tracker_32[frame_number]) +
								 np.float64(landmarks.pt_affdex_tracker_60[frame_number]) + np.float64(
							landmarks.pt_affdex_tracker_62[frame_number])) / 4),
							int((np.float64(landmarks.pt_affdex_tracker_35[frame_number]) + np.float64(
								landmarks.pt_affdex_tracker_33[frame_number]) +
								 np.float64(landmarks.pt_affdex_tracker_61[frame_number]) + np.float64(
										landmarks.pt_affdex_tracker_63[frame_number])) / 4))

				right_eye = (int((np.float64(landmarks.pt_affdex_tracker_36[frame_number]) + np.float64(
					landmarks.pt_affdex_tracker_64[frame_number]) +
								  np.float64(landmarks.pt_affdex_tracker_38[frame_number]) + np.float64(
							landmarks.pt_affdex_tracker_66[frame_number])) / 4),
							 int((np.float64(landmarks.pt_affdex_tracker_37[frame_number]) + np.float64(
								 landmarks.pt_affdex_tracker_65[frame_number]) +
								  np.float64(landmarks.pt_affdex_tracker_39[frame_number]) + np.float64(
										 landmarks.pt_affdex_tracker_67[frame_number])) / 4))

				angle = angle_between((1, 0), (right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])) * 180 / 3.141592
				if left_eye[1] > right_eye[1]:
					angle = -angle
				distance = int(np.linalg.norm((right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])))

				rotate_matrix = cv2.getRotationMatrix2D(center=left_eye, angle=angle, scale=1)
				rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(image.shape[1], image.shape[0]))

				ratio = distance / 100
				rectangle = ((int(left_eye[0] - 75 * ratio), int(left_eye[1] - 70 * ratio)),
							 ((int(left_eye[0] + 175 * ratio), int(left_eye[1] + 180 * ratio))))

				if (len_label > label_idx + 1 and label_csv.iloc[label_idx + 1, 0] * 1000 < landmarks.iloc[frame_number, 0]):  # TimeStamp(msec) csak nem jeleníti meg valamiért
					label_idx = label_idx + 1

				aug_count = 0
				while aug_count < augment_number:
					#print(str(aug_count) + " " + str(augment_number))


					height, width = image.shape[:2]
					if (aug_count == 0):
						delta_x = 0
					else:
						delta_x = (np.random.rand() - 0.5) / 2
					T = np.float32([[1, delta_x, -delta_x * left_eye[1]], [0, 1, 0]])
					img_translation = cv2.warpAffine(rotated_image, T, (width, height))


					cropped_image = img_translation[rectangle[0][1]:rectangle[1][1], rectangle[0][0]:rectangle[1][0]]
					downsampled_image = cv2.resize(cropped_image, (64, 64))




					if (np.float64(label_csv.iloc[label_idx, 1]) == 0):
						label = 0
					else:
						label = 1
					if (np.random.rand() > 0.5):
						downsampled_image = cv2.flip(downsampled_image, 1)
					yield (downsampled_image / 255, label)
					aug_count += 1


			except Exception as e:
				pass
				#print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

			frame_number = frame_number + 1

		count = count + 1
##
# class proxy:
# 	def __init__(self, aug, list):
# 		self.aug = aug
# 		self.list = list
# 		self.gen = dataset_generator(self.aug, self.list)
#
# 	def __call__(self):
# 		return next(self.gen)
#
# 	def __iter__(self):
# 		self.gen = dataset_generator(self.aug, self.list)
# 		return self
#
# 	def __next__(self):
# 		return next(self.gen)

##

# file_negative_positive = dict()
# count = 0
# for file in files:
#
# 	#file = r"3f28a162-f529-4bc4-bbd6-f1ecf92d8b22"
# 	#file = r"0005b896-33ad-4ecf-93b4-dad735ad69b6"
# 	label_csv = pd.read_csv(r"/home/polyamatyas/projects/mosoly/AMFEDPLUS_Distribution/AU Labels/" + file + "-label.csv")
# 	len_label = label_csv.shape[0]
# 	label_idx = 0
#
# 	landmarks = pd.read_csv(r"/home/polyamatyas/projects/mosoly/AMFEDPLUS_Distribution/Landmark Points (labeled videos)/" + file + ".csv")
# 	cap = cv2.VideoCapture(r"/home/polyamatyas/projects/mosoly/AMFEDPLUS_Distribution/Videos - FLV (labeled)/" + file + ".flv")
#
# 	number_of_positives = 0
# 	number_of_negatives = 0
#
# 	frame_number = 0
# 	while cap.isOpened():
# 		frame_exists, image = cap.read()
# 		if not frame_exists:
# 			break
# 		try:
#
# 			if (len_label > label_idx + 1 and label_csv.iloc[label_idx + 1,0] * 1000 < landmarks.iloc[frame_number, 0]): #TimeStamp(msec) csak nem jeleníti meg valamiért
# 				label_idx = label_idx + 1
#
#
# 			if (np.float64(label_csv.iloc[label_idx, 1]) == 0):
# 				number_of_negatives += 1
# 			else:
# 				number_of_positives += 1
#
# 		except:
# 			pass
#
#
#
# 		frame_number = frame_number + 1
#
# 	file_negative_positive[file] = (number_of_negatives, number_of_positives)
#
# 	count = count + 1
# 	print(str(count) + "/" + str(len(files)))
# 	# if count == 2:
# 	# 	break
#
# file = open(r'/home/polyamatyas/projects/mosoly/negative_positive_ratio.pkl', 'wb')
# pickle.dump(file_negative_positive, file)
# file.close()
#


##
#
# file_inputs = dict()
#
# count = 0
# for file in files:
#
# 	#file = r"3f28a162-f529-4bc4-bbd6-f1ecf92d8b22"
# 	#file = r"0005b896-33ad-4ecf-93b4-dad735ad69b6"
# 	label_csv = pd.read_csv(r"/home/polyamatyas/projects/mosoly/AMFEDPLUS_Distribution/AU Labels/" + file + "-label.csv")
# 	len_label = label_csv.shape[0]
# 	label_idx = 0
#
# 	input_images = []
# 	input_labels = []
#
# 	landmarks = pd.read_csv(r"/home/polyamatyas/projects/mosoly/AMFEDPLUS_Distribution/Landmark Points (labeled videos)/" + file + ".csv")
# 	cap = cv2.VideoCapture(r"/home/polyamatyas/projects/mosoly/AMFEDPLUS_Distribution/Videos - FLV (labeled)/" + file + ".flv")
#
# 	frame_number = 0
# 	while cap.isOpened():
# 		frame_exists, image = cap.read()
# 		if not frame_exists:
# 			break
# 		try:
# 			left_eye = (int((np.float64 (landmarks.pt_affdex_tracker_34[frame_number]) + np.float64 (landmarks.pt_affdex_tracker_32[frame_number]) +
# 							 np.float64 (landmarks.pt_affdex_tracker_60[frame_number]) + np.float64 (landmarks.pt_affdex_tracker_62[frame_number])) / 4),
# 						int((np.float64 (landmarks.pt_affdex_tracker_35[frame_number]) + np.float64 (landmarks.pt_affdex_tracker_33[frame_number]) +
# 							 np.float64 (landmarks.pt_affdex_tracker_61[frame_number]) + np.float64 (landmarks.pt_affdex_tracker_63[frame_number])) / 4))
#
# 			right_eye = (int((np.float64 (landmarks.pt_affdex_tracker_36[frame_number]) + np.float64 (landmarks.pt_affdex_tracker_64[frame_number]) +
# 							  np.float64 (landmarks.pt_affdex_tracker_38[frame_number]) + np.float64 (landmarks.pt_affdex_tracker_66[frame_number])) / 4),
# 						 int((np.float64 (landmarks.pt_affdex_tracker_37[frame_number]) + np.float64 (landmarks.pt_affdex_tracker_65[frame_number]) +
# 							  np.float64 (landmarks.pt_affdex_tracker_39[frame_number]) + np.float64 (landmarks.pt_affdex_tracker_67[frame_number])) / 4))
#
# 			angle = angle_between((1, 0), (right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])) * 180 / 3.141592
# 			if left_eye[1] > right_eye[1]:
# 				angle = -angle
# 			distance = int(np.linalg.norm((right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])))
#
# 			rotate_matrix = cv2.getRotationMatrix2D(center=left_eye, angle=angle, scale=1)
# 			rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(image.shape[1], image.shape[0]))
#
# 			ratio = distance / 100
# 			rectangle = ((int(left_eye[0] - 75 * ratio), int(left_eye[1] - 70 * ratio)),
# 						 ((int(left_eye[0] + 175 * ratio), int(left_eye[1] + 180 * ratio))))
#
# 			cropped_image = rotated_image[rectangle[0][1]:rectangle[1][1], rectangle[0][0]:rectangle[1][0]]
# 			downsampled_image = cv2.resize(cropped_image, (64, 64))
#
#
#
# 			if (len_label > label_idx + 1 and label_csv.iloc[label_idx + 1,0] * 1000 < landmarks.iloc[frame_number, 0]): #TimeStamp(msec) csak nem jeleníti meg valamiért
# 				label_idx = label_idx + 1
#
# 			if (np.float64(label_csv.iloc[label_idx, 1]) == 0):
# 				input_labels.append(0)
# 			else:
# 				input_labels.append(1)
# 			input_images.append(downsampled_image / 255)
#
#
# 		except Exception as e:
# 			#print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
# 			pass
#
#
#
# 		frame_number = frame_number + 1
#
# 	file_inputs[file] = (input_images, input_labels)
#
# 	count = count + 1
# 	print(str(count) + "/" + str(len(files)))
# 	# if count == 2:
# 	# 	break
#
#
#
# file = open(r'/home/polyamatyas/projects/mosoly/data.pkl', 'wb')
# pickle.dump(file_inputs, file, protocol=4)
# file.close()

##
def load_inputs_from_file():
	global training_images
	global training_labels
	global validation_images
	global validation_labels
	global test_images
	global test_labels

	file = open(r'/home/polyamatyas/projects/mosoly/data.pkl', 'rb')
	file_inputs = pickle.load(file)
	file.close()

	training_images = []
	training_labels = []

	test_images = []
	test_labels = []

	validation_images = []
	validation_labels = []
	for file in training_files:
		training_images.extend(file_inputs[file][0])
		training_labels.extend(file_inputs[file][1])


	for file in test_files:
		test_images.extend(file_inputs[file][0])
		test_labels.extend(file_inputs[file][1])


	for file in validation_files:
		validation_images.extend(file_inputs[file][0])
		validation_labels.extend(file_inputs[file][1])

	file_inputs = 0



	training_images, training_labels = np.array(training_images), np.array(training_labels)
	validation_images, validation_labels = np.array(validation_images), np.array(validation_labels)
	test_images, test_labels = np.array(test_images), np.array(test_labels)



##
model = models.Sequential()
model.add(layers.Conv2D(64, (7, 7), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.Dropout(0.3))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(50))

model.summary()

##
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=10,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True
)


checkpoint_filepath = r'/home/polyamatyas/projects/mosoly/models/checkpoint_{epoch:02d}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    save_best_only=False)



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

raw_train_generator = lambda: dataset_generator(4,training_files)
raw_val_generator = lambda: dataset_generator(1,validation_files)

tr_generator = tf.data.Dataset.from_generator(raw_train_generator, (tf.float64, tf.int32),    (tf.TensorShape([64,64,3]), tf.TensorShape([])))
val_generator = tf.data.Dataset.from_generator(raw_val_generator,  (tf.float64, tf.int32),    (tf.TensorShape([64,64,3]), tf.TensorShape([])))

history = model.fit_generator(tr_generator.batch(1), epochs=20,
                    validation_data=val_generator.batch(1),
					callbacks=[early_stopping_callback, model_checkpoint_callback],
					use_multiprocessing=True,
					class_weight=class_weight)


# load_inputs_from_file()
# history = model.fit(training_images,
# 					training_labels,
# 					epochs=20,
#                     validation_data=(validation_images, validation_labels),
# 					callbacks=[early_stopping_callback, model_checkpoint_callback],
# 					class_weight=class_weight)





##


# path = r'/home/polyamatyas/projects/mosoly/models/checkpoint_15.hdf5'
# savedModel = models.load_model(path)
# pred = savedModel(test_images[:1000])
# pred = np.argmax(pred, axis = 1)
#
# wrong_pred = []
# for i in range(len(pred)):
# 	#if pred[i] != test_labels[i]:
# 	#	wrong_pred.append(test_images[i]*255)
# 	print("predicted: " + str(pred[i]) + " Real value: " + str(test_labels[i]) + " Index: " + str(i))
#
#
# a = dataset_generator(4, training_files)
# for i in range(1000):
# 	b = next(a)
# 	print(i , " " , b[1])
