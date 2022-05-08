##
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

##

fig, ax = plt.subplots()
x, ysin, ycos = [], [], []
ln1, = plt.plot([], [], 'ro')
ln2, = plt.plot([], [], 'm*')

##

def init():
	ax.set_xlim(0, 2 * np.pi)
	ax.set_ylim(-1, 1)


def update(i):
	x.append(i)
	ysin.append(np.sin(i))
	ycos.append(np.cos(i))
	ln1.set_data(x, ysin)
	ln2.set_data(x, ycos)


ani = FuncAnimation(fig, update, np.linspace(0, 2 * np.pi, 64), init_func=init)
plt.show()

writer = PillowWriter(fps=25)
ani.save("demo_sine.gif", writer=writer)
