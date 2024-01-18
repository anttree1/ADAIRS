import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 4, 5, 6])
X, Y = np.meshgrid(x, y)
Z = np.array([[0.51730769, 0.67596154, 0.81538462, 0.78173077, 0.93269231, 0.88461538],
              [0.58653846, 0.99038462, 1., 0.93269231, 0.99038462, 0.98365385],
              [0.75, 0.69230769, 0.70192308, 0.82692308, 0., 0.82692308],
              [0.51923077, 0.50961538, 0.55769231, 0.48076923, 0.46153846, 0.44230769],
              [0.29807692, 0.33653846, 0.30769231, 0.34615385, 0.32692308, 0.35576923],
              [0.17307692, 0., 0.02884615, 0.04807692, 0.05769231, 0.01923077]]
             )

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, projection='3d')

dx = dy = 0.4
dz = Z.flatten()
cmap = plt.cm.get_cmap('Blues')
ax.bar3d(X.flatten(), Y.flatten(), np.zeros_like(Z.flatten()), dx, dy, dz, color=cmap(dz))


ax.set_xlabel('D')
ax.set_ylabel('L')
ax.set_zlabel('Z')

sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array(Z)
fig.colorbar(sm)
ax.set_xticks(x + 0.015)
ax.set_xticklabels(x)
ax.set_yticks(y + 0.05)
ax.set_yticklabels(y)
ax.set_zlim(0, np.max(dz))

plt.show()
