import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'GTK3Agg', etc.

a_vals = np.linspace(0.001, 0.999, 100)
y_vals = np.linspace(0, 1, 100)
A, Y = np.meshgrid(a_vals, y_vals)
Z = -Y * np.log(A) - (1 - Y) * np.log(1 - A)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, Y, Z, cmap='viridis')
ax.set_xlabel('a (prediction)')
ax.set_ylabel('y (target)')
ax.set_zlabel('C(a, y) (Cross-entropy cost)')
plt.show()