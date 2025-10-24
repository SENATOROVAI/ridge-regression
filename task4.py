import numpy as np
import matplotlib.pyplot as plt

w1, w2 = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
f = w1**2 + 4*w2**2
plt.contour(w1, w2, f, levels=[1, 2, 4, 8])
plt.axis('equal')
plt.show()
