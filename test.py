import numpy as np
from numpy.random import normal as normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib
nfr = 30
fps = 10
xs = []
ys = []
zs = []
ss = np.arange(1, nfr, 0.5)
for s in ss:
    xs.append(normal(50, s, 1))
    ys.append(normal(50, s, 1))
    zs.append(normal(50, s, 1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sct, = ax.plot([], [], [], "o", markersize=2)


def update(ifrm, xa, ya, za):
    sct.set_data(xa[ifrm], ya[ifrm])
    sct.set_3d_properties(za[ifrm])


ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)
ani = animation.FuncAnimation(fig, update, nfr, fargs=(xs, ys, zs), interval=1000/fps)
plt.show()
