# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""## Подготовка данных"""

boston = pd.read_csv('/content/boston.csv')

X = boston[['LSTAT', 'RM', 'PTRATIO', 'INDUS']]
y = boston.MEDV

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""## Линейная регрессия

### Ordinary Least Squares
"""

# %%

"""#### Метод градиентного спуска

собственный класс
"""

class RidgeGD():
  def __init__(self, alpha = 0.0001):
    self.thetas = None
    self.loss_history = []
    self.alpha = alpha

  def add_ones(self, x):
    return np.c_[np.ones((len(x), 1)), x]

  def objective(self, x, y, thetas, n):
    return (np.sum((y - self.h(x, thetas)) ** 2) + self.alpha * np.dot(thetas, thetas)) / (2 * n)

  def h(self, x, thetas):
    return np.dot(x, thetas)

  def gradient(self, x, y, thetas, n):
    return (np.dot(-x.T, (y - self.h(x, thetas))) + (self.alpha * thetas)) / n

  def fit(self, x, y, iter = 20000, learning_rate = 0.05):
    x, y = x.copy(), y.copy()
    x = self.add_ones(x)

    thetas, n = np.zeros(x.shape[1]), x.shape[0]

    loss_history = []

    for i in range(iter):
      loss_history.append(self.objective(x, y, thetas, n))
      grad = self.gradient(x, y, thetas, n)
      thetas -= learning_rate * grad

    self.thetas = thetas
    self.loss_history = loss_history

  def predict(self, x):
    x = x.copy()
    x = self.add_ones(x)
    return np.dot(x, self.thetas)

ridge_gd = RidgeGD(alpha = 0.0001)

ridge_gd.fit(X_train, y_train, iter = 50000, learning_rate = 0.01)

y_pred_train = ridge_gd.predict(X_train)
y_pred_test = ridge_gd.predict(X_test)

print('train: ' + str(root_mean_squared_error(y_train, y_pred_train)))
print('test: ' + str(root_mean_squared_error(y_test, y_pred_test)))

"""класс sklearn"""

from sklearn.linear_model import SGDRegressor

ridge_gd = SGDRegressor(loss = 'squared_error',
                        penalty = 'l2',
                        alpha = 0.0001,
                        max_iter = 50000,
                        learning_rate = 'constant',
                        eta0 = 0.001,
                        random_state = 42)

ridge_gd.fit(X_train, y_train)

y_pred_train = ridge_gd.predict(X_train)
y_pred_test = ridge_gd.predict(X_test)

print('train: ' + str(root_mean_squared_error(y_train, y_pred_train)))
print('test: ' + str(root_mean_squared_error(y_test, y_pred_test)))

"""#### Выбор $\alpha$"""

from sklearn.linear_model import RidgeCV

# укажем параметры регуляризации alpha, которые хотим протестировать
# дополнительно укажем количество частей (folds), параметр cv,
# на которое нужно разбить данные при оценке качества модели
ridge_cv = RidgeCV(alphas = [0.1, 1.0, 10], cv = 10)
ridge_cv.fit(X_train, y_train)

# выведем оптимальный параметр и достигнутое качество
ridge_cv.alpha_, ridge_cv.best_score_




