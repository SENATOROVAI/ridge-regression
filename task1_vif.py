import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import numpy as np

# Создаем данные с мультиколлинеарностью
np.random.seed(42)
n = 100
x1 = np.random.normal(0, 1, n)
x2 = x1 + np.random.normal(0, 0.1, n)  # Сильно коррелирован с x1
x3 = np.random.normal(0, 1, n)         # Независимый признак
x4 = x3 * 2 + np.random.normal(0, 0.2, n)  # Коррелирован с x3

X = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})
X_with_const = add_constant(X)

# Вычисляем VIF для каждого признака
vif_data = pd.DataFrame()
vif_data['Variable'] = X_with_const.columns
vif_data['VIF'] = [variance_inflation_factor(X_with_const.values, i) 
                   for i in range(X_with_const.shape[1])]

print(vif_data)
