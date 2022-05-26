##
import sklearn
from imutils import face_utils
import numpy as np
import imutils
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import mediapipe as mp
from sklearn import metrics
import gc
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import pickle
try:
	import dlib
	from imgaug import augmenters as iaa
except Exception as e:
	pass

import dlib
from imgaug import augmenters as iaa


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
seq = iaa.Sequential([
	iaa.Fliplr(0.5),  # horizontally flip 50% of the images
	iaa.Affine(shear={"x": (-30, 30)}),
		iaa.GaussianBlur(sigma=(0, 1.0))  # blur images with a sigma of 0 to 1
])

##
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/polyamatyas/projects/mosoly/shape_predictor_68_face_landmarks(1).dat")


##
# Read raw images and labels
def read_from_files():
	file_images = open('/home/polyamatyas/projects/mosoly/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images.txt', 'r')
	file_labels = open('/home/polyamatyas/projects/mosoly/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt', 'r')
	count = 0

	raw_images = []
	raw_labels = []

	while True:
		line = file_images.readline().strip()
		label_line = file_labels.readline().strip()
		if not line:
			break

		raw_labels.append(int(label_line[0]))
		# print(line)
		raw_images.append(cv2.imread("/home/polyamatyas/projects/mosoly/GENKI-R2009a/files/" + line))

		count = count + 1

	file_images.close()
	file_labels.close()
	return raw_images, raw_labels


##
# |######################################
# Augment images
# |#####################################
def augment(raw_images, raw_labels, augment_number):
	test_images = []
	test_labels = []

	count = 0
	for original_image, label in zip(raw_images, raw_labels):
		batch = [original_image for i in range(augment_number)]
		augmented_images_batch = seq(images=batch)
		for x in augmented_images_batch:
			test_images.append(x)
			test_labels.append(label)

		# for i in range(10):
		#   cv2_imshow(test_images[count * 10 + i])

		count = count + 1

	# if count == 10:
	#   break
	return test_images, test_labels


##
# |#####################################
# Make images into inputs
# |#####################################
def make_into_inputs(test_images, test_labels):
	input_images = []
	input_labels = []

	count = 0
	mp_face_mesh = mp.solutions.face_mesh
	for original_image, label in zip(test_images, test_labels):

		image = original_image
		image = imutils.resize(image, width=500)

		with mp_face_mesh.FaceMesh(
				max_num_faces=1,
				# refine_landmarks=True,
				min_detection_confidence=0.5,
				min_tracking_confidence=0.5) as face_mesh:
			mp_face_mesh = mp.solutions.face_mesh
			shape_y, shape_x = image.shape[:2]
			landmark_scaling = np.array([shape_x, shape_y, shape_x])

			results = face_mesh.process(image)
			if results.multi_face_landmarks:
				for face in results.multi_face_landmarks:
					landmarks = [[l.x, l.y, l.z] for l in face.landmark]

				landmarks = np.array(landmarks) * landmark_scaling
				# for idx, l in enumerate(landmarks):
				#   cv2.circle(image, (int(l[0]), int(l[1])), 1, (0, 0, 0), -1)

				left_eye = (landmarks[33] + landmarks[246] + landmarks[161] + landmarks[160] + landmarks[159] +
							landmarks[
								158] + landmarks[157] + landmarks[173] + landmarks[133] + landmarks[155] + landmarks[
								154] + landmarks[
								153] + landmarks[145] + landmarks[144] + landmarks[163] + landmarks[7]) // 16
				right_eye = (landmarks[362] + landmarks[398] + landmarks[384] + landmarks[385] + landmarks[386] +
							 landmarks[
								 387] + landmarks[388] + landmarks[466] + landmarks[263] + landmarks[249] + landmarks[
								 390] + landmarks[
								 373] + landmarks[374] + landmarks[380] + landmarks[381] + landmarks[382]) // 16

				# for x in landmarks:
				# 	cv2.circle(image, (int(x[0]), int(x[1])), 2, (0, 255, 255), -1)
				#
				# for i in [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]:
				# 	cv2.circle(image, (int(landmarks[i][0]), int(landmarks[i][1])), 2, (254, 0, 0), -1)
				#
				# cv2.circle(image, (int(left_eye[0]), int(left_eye[1])), 2, (255, 255, 0), -1)
				# cv2.circle(image, (int(right_eye[0]), int(right_eye[1])), 2, (255, 255, 0), -1)
			# cv2_imshow(image)
			else:
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				# detect faces in the grayscale image
				rects = detector(gray, 1)

				# loop over the face detections
				for (i, rect) in enumerate(rects):
					# determine the facial landmarks for the face region, then
					# convert the facial landmark (x, y)-coordinates to a NumPy
					# array
					shape = predictor(gray, rect)
					shape = face_utils.shape_to_np(shape)
					# convert dlib's rectangle to a OpenCV-style bounding box
					# [i.e., (x, y, w, h)], then draw the face bounding box
					# (x, y, w, h) = face_utils.rect_to_bb(rect)
					# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

					# show the face number
					# cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
					# cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

					# loop over the (x, y)-coordinates for the facial landmarks
					# and draw them on the image
					# for (x, y) in shape:
					# 	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

					left_eye = (shape[36] + shape[37] + shape[38] + shape[39] + shape[40] + shape[41]) // 6
					right_eye = (shape[42] + shape[43] + shape[44] + shape[45] + shape[46] + shape[47]) // 6
		# cv2.circle(image, (left_eye[0], left_eye[1]), 1, (0, 255, 0), -1)
		# cv2.circle(image, (right_eye[0], right_eye[1]), 1, (0, 255, 0), -1)

		# show the output image with the face detections + facial landmarks
		# cv2_imshow(image)
		# cv2.waitKey(0)

		# |###########################################################################
		image = original_image
		image = imutils.resize(image, width=500)
		# |###########################################################################

		# detect faces in the grayscale image

		left_eye_desired_position = (150, 150)

		height, width = image.shape[:2]

		T = np.float32(
			[[1, 0, left_eye_desired_position[0] - left_eye[0]], [0, 1, left_eye_desired_position[1] - left_eye[1]]])
		img_translation = cv2.warpAffine(image, T, (width, height))

		angle = angle_between((1, 0), (right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])) * 180 / 3.141592
		if left_eye[1] > right_eye[1]:
			angle = -angle

		rotate_matrix = cv2.getRotationMatrix2D(center=left_eye_desired_position, angle=angle,
												scale=100 / np.linalg.norm(right_eye - left_eye))
		rotated_image = cv2.warpAffine(src=img_translation, M=rotate_matrix, dsize=(image.shape[1], image.shape[0]))

		rectangle = ((75, 80), (325, 330))
		# cv2.rectangle(rotated_image, rectangle[0], rectangle[1], (0, 255, 0), 1)
		# cv2_imshow(rotated_image)

		cropped_image = rotated_image[rectangle[0][1]:rectangle[1][1], rectangle[0][0]:rectangle[1][0]]
		# cv2_imshow(cropped_image)

		downsampled_image = cv2.resize(cropped_image, (64, 64))
		input_images.append(downsampled_image / 255)
		input_labels.append(label)

		# |###########################################################################

		count = count + 1
		if count % 1000 == 0:
			print(count)
	return input_images, input_labels


##
# raw_images, raw_labels = read_from_files()
# augmented_images, augmented_labels = augment(raw_images, raw_labels, 20)
#
# original_input_images, original_input_labels = make_into_inputs(raw_images, raw_labels)
# augmented_input_images, augmented_input_labels = make_into_inputs(augmented_images, augmented_labels)
# import pickle
#
# file = open('GENKI_inputs_20_augment.pkl', 'wb')
# pickle.dump(original_input_images, file)
# pickle.dump(original_input_labels, file)
# pickle.dump(augmented_input_images, file)
# pickle.dump(augmented_input_labels, file)
# file.close()


##
file = open('GENKI_inputs_20_augment.pkl', 'rb')
original_input_images = pickle.load(file)
original_input_labels = pickle.load(file)
augmented_input_images = pickle.load(file)
augmented_input_labels = pickle.load(file)
file.close()


##

# # Function to compute Nth digit
# # from right in base B
# def nthDigit(a, n, b):
# 	for i in range(1, n):
# 		a = a // b
# 	return a % b


##
def generate_inputs(original_input_images, original_input_labels, augmented_input_images, augmented_input_labels, augment_rate):
	training_rate, validation_rate, test_rate = 0.8, 0.1, 0.1
	permutation = np.random.permutation(len(original_input_images))

	full_augment_rate = len(augmented_input_images) // len(original_input_images)

	training_images, training_labels = [], []
	validation_images, validation_labels = [], []
	test_images, test_labels = [], []

	for i in range(0, len(original_input_images)):
		number = permutation[i]

		if (i < len(original_input_images) * training_rate):
			training_images.append(original_input_images[number])
			training_labels.append(original_input_labels[number])

			training_images.extend(augmented_input_images[number * full_augment_rate: number * full_augment_rate + augment_rate])
			training_labels.extend(augmented_input_labels[number * full_augment_rate: number * full_augment_rate + augment_rate])

		elif (i < len(original_input_images) * (training_rate + validation_rate)):
			validation_images.append(original_input_images[number])
			validation_labels.append(original_input_labels[number])
		else:
			test_images.append(original_input_images[number])
			test_labels.append(original_input_labels[number])

	training_images, training_labels = np.array(training_images), np.array(training_labels)
	validation_images, validation_labels = np.array(validation_images), np.array(validation_labels)
	test_images, test_labels = np.array(test_images), np.array(test_labels)

	return training_images, training_labels, validation_images, validation_labels, test_images, test_labels


##
# for i in range(1, 4):
# 	for a in range(pow(3, i)):
# 		for number_of_teaching in range(10):
#
# 			training_images, training_labels, validation_images, validation_labels, test_images, test_labels = generate_inputs(original_input_images, original_input_labels, augmented_input_images, augmented_input_labels, 5)
#
# 			model = models.Sequential()
# 			model_id = ""
# 			for n in range(1, i + 1):
# 				convolution_case = nthDigit(a, n, 3)
# 				conv_shape = 3 + convolution_case * 2  # 0:3x3 1:5x5 2:7x7
# 				model_id += str(conv_shape)
# 				if (n == 1):
# 					# print(conv_shape, " input")
# 					model.add(layers.Conv2D(32, (conv_shape, conv_shape), activation='relu', input_shape=(64, 64, 3)))
# 				else:
# 					# print(conv_shape)
# 					model.add(layers.Conv2D(32, (conv_shape, conv_shape), activation='relu'))
# 				model.add(layers.Dropout(0.2))
# 				model.add(layers.MaxPooling2D((2, 2)))
#
# 			model.add(layers.Flatten())
# 			model.add(layers.Dense(30))
# 			model.add(layers.Dense(2))
#
# 			print(model_id, number_of_teaching)
# 			#model.summary()
#
# 			early_stopping_callback = tf.keras.callbacks.EarlyStopping(
# 				monitor="val_loss",
# 				min_delta=0,
# 				patience=15,
# 				verbose=0,
# 				mode="auto",
# 				baseline=None,
# 				restore_best_weights=True
# 			)
#
# 			checkpoint_filepath = r'/home/polyamatyas/projects/mosoly/models_genki_10/' + model_id + r'_' + str(number_of_teaching) + r'_{epoch}.hdf5'
# 			model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
# 				filepath=checkpoint_filepath,
# 				save_weights_only=False,
# 				monitor='val_accuracy',
# 				save_best_only=False)
#
# 			model.compile(optimizer='adam',
# 						  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
# 						  metrics=['accuracy'])
#
# 			history = model.fit(training_images,
# 								training_labels,
# 								epochs=30,
# 								validation_data=(validation_images, validation_labels),
# 								callbacks=[early_stopping_callback])#, model_checkpoint_callback])
#
# 			file = open(r'/home/polyamatyas/projects/mosoly/models_genki_10/' + model_id + '_' + str(number_of_teaching) + r'_history.pkl', 'wb')
# 			pickle.dump(history.history, file)
# 			file.close()
#
# 			tf.keras.backend.clear_session()
# 			del model
# 			gc.collect()
#
#
#

##
# for i in range(11):
# 	for j in range(10):
training_images, training_labels, validation_images, validation_labels, test_images, test_labels = generate_inputs(
	original_input_images, original_input_labels, augmented_input_images, augmented_input_labels, 5)

model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (5, 5), activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(30))
model.add(layers.Dense(2))

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
	monitor="val_loss",
	min_delta=0,
	patience=15,
	verbose=0,
	mode="auto",
	baseline=None,
	restore_best_weights=True	)

#checkpoint_filepath = r'/home/polyamatyas/projects/mosoly/genki_models_augment/' + str(i) + r'_' + str(j) + r'_{epoch}.hdf5'
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
# 	filepath=checkpoint_filepath,
# 	save_weights_only=False,
# 	monitor='val_accuracy',
# 	save_best_only=False)

model.compile(optimizer='adam',
			  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			  metrics=['accuracy'])

history = model.fit(training_images,
					training_labels,
					epochs=30,
					validation_data=(validation_images, validation_labels),
					callbacks=[early_stopping_callback])#, model_checkpoint_callback])

# file = open(r'/home/polyamatyas/projects/mosoly/genki_models_augment_535/' + str(i) + '_' + str(j) + r'_history.pkl', 'wb')
# pickle.dump(history.history, file)
# file.close()
# print("augment: ",str(i), "n: ", str(j))
# tf.keras.backend.clear_session()
# del model
# gc.collect()
#

##
# model_name = "555"
# # file = open(r'/home/polyamatyas/projects/mosoly/models/' + model_name + 'history.pkl', 'rb')
# file = open(r'/home/polyamatyas/projects/mosoly/models_genki/3history.pkl', 'rb')
# history = pickle.load(file)
# file.close()
#
# plt.plot(history['accuracy'], label='accuracy')
# plt.plot(history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.figtext(.15, .15, model_name)
# plt.show()
##

#GENKI on GENKI
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# 4000 -----  loss: 0.2665 - accuracy: 0.8440


genki_to_genki_pred = model.predict(test_images)
genki_to_genki_pred_class = [0 if x[0] > x[1] else 1 for x in genki_to_genki_pred]

genki_to_genki_correctness = [[0,0], [0,0]]
for label, pred in zip(test_labels, genki_to_genki_pred_class):
	genki_to_genki_correctness[label][pred] += 1
# [176, 11]
# [10, 203]

##

missclassification_pictures = [[], []]
for i in range(len(test_images)):
	if (test_labels[i] != genki_to_genki_pred_class[i]):
		missclassification_pictures[test_labels[i]].append(test_images[i])

##
#roc
def softmax(vec):
	exponential = np.exp(vec)
	probabilities = exponential / np.sum(exponential)
	return probabilities

probabilities = []

for x in genki_to_genki_pred:
	probabilities.append(softmax(x))

y_pred_proba = np.array(probabilities)[:,1]
fpr, tpr, _ = sklearn.metrics.roc_curve(test_labels,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()