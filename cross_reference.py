##
import numpy as np
import imutils
import cv2
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

import matplotlib.pyplot as plt

import mediapipe as mp
import pandas as pd
import os
import pickle
import gc

##

models = ["3", "5", "7",
		  "33", "35", "37", "53", "55", "57", "73", "75", "77",
		  "333", "335", "337", "353", "355", "357", "373", "375", "377",
		  "533", "535", "537", "553", "555", "557", "573", "575", "577",
		  "733", "735", "737", "753", "755", "757", "773", "775", "777"]

history_list = []

for x in models:
	file = open(r'/home/polyamatyas/projects/mosoly/models/' + x + 'history.pkl', 'rb')
	history = pickle.load(file)
	file.close()

	history_list.extend([(acc, i, x) for i, acc in enumerate(history['val_accuracy'])])

val_acc_list = [x[0] for x in history_list]
max_index = val_acc_list.index(max(val_acc_list))

# 370 for genki: (0.9575, 8, '337')
# 452 for amfed: (0.8913622, 3, '735')

##

def amfed_load_inputs_from_file():
	dir_list = os.listdir(r"/home/polyamatyas/projects/mosoly/AMFEDPLUS_Distribution/Landmark Points (labeled videos)")
	files = [text.split(".")[0] for text in dir_list if (os.path.getsize(
		r"/home/polyamatyas/projects/mosoly/AMFEDPLUS_Distribution/Landmark Points (labeled videos)/" + text) != 0)]

	file = open(r'/home/polyamatyas/projects/mosoly/data.pkl', 'rb')
	file_inputs = pickle.load(file)
	file.close()

	input_images = []
	input_labels = []

	for file in files:
		input_images.extend(file_inputs[file][0])
		input_labels.extend(file_inputs[file][1])


	input_images, input_labels = np.array(input_images), np.array(input_labels)

	return input_images, input_labels
##
def genki_load_inputs_from_file():
	file = open(r'/home/polyamatyas/projects/mosoly/GENKI_inputs.pkl', 'rb')
	original_input_images = pickle.load(file)
	original_input_labels = pickle.load(file)
	#augmented_input_images = pickle.load(file)
	#augmented_input_labels = pickle.load(file)
	file.close()

	original_input_images, original_input_labels = np.array(original_input_images), np.array(original_input_labels)
	return original_input_images, original_input_labels
##

genki_model_path = r'/home/polyamatyas/projects/mosoly/models_genki/337_08.hdf5'
amfed_model_path = r'/home/polyamatyas/projects/mosoly/models/735_03.hdf5'

genki_model = tf.keras.models.load_model(genki_model_path)
amfed_model = tf.keras.models.load_model(amfed_model_path)
genki_model.summary()
amfed_model.summary()

amfed_images, amfed_labels = amfed_load_inputs_from_file()
genki_images, genki_labels = genki_load_inputs_from_file()

##
# GENKI dataset on AMFED model

loss, acc = amfed_model.evaluate(genki_images, genki_labels, verbose=2)
# 4000 -----  loss: 0.2665 - accuracy: 0.8440


amfed_to_genki_pred = amfed_model.predict(genki_images)
amfed_to_genki_pred_class = [0 if x[0] > x[1] else 1 for x in amfed_to_genki_pred]

amfed_to_genki_correctness = [[0,0], [0,0]]
for label, pred in zip(genki_labels, amfed_to_genki_pred_class):
	amfed_to_genki_correctness[label][pred] += 1
# [1605, 233]
# [391, 1771]

##

# AMFED dataset on GENKI model

#loss, acc = genki_model.evaluate(amfed_images, amfed_labels, verbose=2)
# 229854 ----- loss: 0.2407 - accuracy: 0.8166

amfed_to_genki_pred_class = []

count = 0
for i in range(len(amfed_images) // 1000):

	genki_to_amfed_pred = genki_model.predict(amfed_images[i * 1000: (i + 1) * 1000])
	amfed_to_genki_pred_class.extend([0 if p[0] > p[1] else 1 for p in genki_to_amfed_pred])
	print(str(count), '/', str(len(amfed_images) // 1000))
	count +=1

genki_to_amfed_correctness = [[0,0], [0,0]]
for label, pred in zip(amfed_labels, amfed_to_genki_pred_class):
	genki_to_amfed_correctness[label][pred] += 1
# [163601, 2883]
# [39272, 23244]