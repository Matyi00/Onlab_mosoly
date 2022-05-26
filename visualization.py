##
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pickle
import statistics
import math

##

fig = plt.figure(figsize=(15, 8))
axis = plt.axes()
line, = axis.plot([], [], lw = 2)

##
models = ["3", "5", "7",
		  "33", "35", "37", "53", "55", "57", "73", "75", "77",
		  "333", "335", "337", "353", "355", "357", "373", "375", "377",
		  "533", "535", "537", "553", "555", "557", "573", "575", "577",
		  "733", "735", "737", "753", "755", "757", "773", "775", "777"]


def init():

	# line.ylabel('Accuracy')
	pass



def update(i):
	#ln1.clear()

	axis.clear()

	axis.set_ylim(0.5, 1)
	for j in range(10):
		file = open(r'/home/polyamatyas/projects/mosoly/models_genki_10/' + models[i] + '_' + str(j) + '_history.pkl', 'rb')
		history = pickle.load(file)
		file.close()
		#print(history['accuracy'])

		axis.plot(history['accuracy'], color=(0,1,0))
		axis.plot(history['val_accuracy'], color=(1,0,0))
	# axis.plot(i,i)

	for txt in fig.texts:
		txt.set_visible(False)
	fig.text(.15, .15, 'model: ' + models[i], fontsize=20)


	return line,

##
ani = FuncAnimation(fig, update, frames=39, init_func=init)
plt.show()

writer = PillowWriter(fps=5)
ani.save("genki_models_10.gif", writer=writer)

##
#calculate maxes

maxes = dict()
for x in range(11):
	max_val = []
	for i in range(10):
		#file = open(r'/home/polyamatyas/projects/mosoly/models_genki_10/' + x + '_' + str(i) + '_history.pkl', 'rb')
		#file = open(r'/home/polyamatyas/projects/mosoly/genki_models_augment/' + str(x) + '_' + str(i) + '_history.pkl', 'rb')
		file = open(r'/home/polyamatyas/projects/mosoly/genki_models_augment_535/' + str(x) + '_' + str(i) + '_history.pkl',
					'rb')

		history = pickle.load(file)
		file.close()

		max_val.append(max(history["val_accuracy"]))
	maxes[x] = max_val

#calculate average and variance
average = dict()
for key, value in maxes.items():
	average[key] = statistics.mean(value)

std_deviation = dict()
for key, value in maxes.items():
	mse = sum([(x-average[key]) * (x-average[key]) for x in value]) / (len(value) - 1)
	std_deviation[key] = math.sqrt(mse)

x = []
y = []
for key in maxes:
	x.append(average[key])
	y.append(key)

plt.plot(y, x)
plt.xlabel('Number of augmention')
plt.ylabel('Average of validation accuracy')
plt.show()

##
fig = plt.figure(figsize=(30, 18))
# for i, x in enumerate(models):
# 	file = open(r'/home/polyamatyas/projects/mosoly/models_genki_10/' + x + 'history.pkl', 'rb')
# 	history = pickle.load(file)
# 	file.close()
#
# 	plt.plot(history['accuracy'], color = (0.8 - i * 0.015, 0.1, 0.2 + i * 0.02))
# 	plt.plot(history['val_accuracy'], color = (0.1, 0.2 + i * 0.02, 0.2))

for i in range(7,8):
	file = open(r'/home/polyamatyas/projects/mosoly/models_genki_10/77_' + str(i) + '_history.pkl', 'rb')
	history = pickle.load(file)
	file.close()
	print(i)

	plt.plot(history['accuracy'], color = (0.8 - i * 0.015, 0.1, 0.2 + i * 0.02), label=str(i))
	#plt.plot(history['val_accuracy'], color = (0.1, 0.2 + i * 0.02, 0.2))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='lower right')
#plt.figtext(.15, .15, model_name)
plt.show()