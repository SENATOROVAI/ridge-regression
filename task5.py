import numpy as np
XTX = np.array([[ЧИСЛА], [ЧИСЛА]])
XTy = np.array([ЧИСЛА])

np.linalg.обратная(XTX + lam * np.eye(2)) @ XTy
