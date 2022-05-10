##
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pickle

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
	file = open(r'/home/polyamatyas/projects/mosoly/models/' + models[i] + 'history.pkl', 'rb')
	history = pickle.load(file)
	file.close()

	#ln1.clear()

	axis.clear()
	axis.set_ylim(0.5, 1)
	axis.plot(history['accuracy'], label='accuracy')
	axis.plot(history['val_accuracy'], label='val_accuracy')
	# axis.plot(i,i)

	for txt in fig.texts:
		txt.set_visible(False)
	fig.text(.15, .15, 'model: ' + models[i], fontsize=20)

	return line,

##
ani = FuncAnimation(fig, update, frames=38, init_func=init)
plt.show()

writer = PillowWriter(fps=5)
ani.save("amfed_models.gif", writer=writer)



##
fig = plt.figure(figsize=(30, 18))
for i, x in enumerate(models):
	file = open(r'/home/polyamatyas/projects/mosoly/models/' + x + 'history.pkl', 'rb')
	history = pickle.load(file)
	file.close()

	plt.plot(history['accuracy'], color = (0.8 - i * 0.015, 0.1, 0.2 + i * 0.02))
	plt.plot(history['val_accuracy'], color = (0.1, 0.2 + i * 0.02, 0.2))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.65, 1])
plt.legend(loc='lower right')
#plt.figtext(.15, .15, model_name)
plt.show()