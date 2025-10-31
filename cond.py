import numpy as np
X = np.array([[1, 2], [2, 4.1]])
cond = np.linalg.cond(X)
print(cond)
