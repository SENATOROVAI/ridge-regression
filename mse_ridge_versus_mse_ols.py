from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Генерация данных с коррелированными признаками
np.random.seed(42)
n_samples, n_features = 50, 20
X = np.random.randn(n_samples, n_features)
# Создаем корреляцию между признаками
X = X + np.random.randn(n_features) * 0.5
y = X.dot(np.random.randn(n_features)) + np.random.randn(n_samples) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Обучение моделей
ols = LinearRegression()
ols.fit(X_train, y_train)

ridge = Ridge(alpha=10.0)  # Подобранный параметр
ridge.fit(X_train, y_train)

# Предсказания и оценка
mse_ols = mean_squared_error(y_test, ols.predict(X_test))
mse_ridge = mean_squared_error(y_test, ridge.predict(X_test))

print(f"MSE OLS: {mse_ols:.4f}")
print(f"MSE Ridge: {mse_ridge:.4f}")
print(f"Ridge лучше на {((mse_ols - mse_ridge)/mse_ols*100):.1f}%")
