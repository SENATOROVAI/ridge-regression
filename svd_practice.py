# %% [markdown]
# # SVD в гребневой (Ridge) регрессии 
# Этот ноутбук включает:
# - Теорию о сингулярном разложении и Ridge-регрессии
# - Визуализации влияния λ на коэффициенты и сингулярные значения

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# %% [markdown]
# ##  Теоретическая часть
# **SVD-разложение:**
# 
# $X = UΣV^T$
# 
# - U — ортонормированные столбцы (наблюдения)
# - Σ — диагональная матрица сингулярных значений
# - V — ортонормированные столбцы (признаки)
# 
# **Ridge-регрессия:**
# $w_{ridge} = (X^T X + λI)^{-1} X^T y$
# 
# Через SVD:
# $w_{ridge} = V D U^T y$, где $D = diag(\frac{σ_i}{σ_i^2 + λ})$
# 
# → малые σ подавляются, повышая устойчивость решения.

# %%

url = 'https://raw.githubusercontent.com/SENATOROVAI/ridge-regression/refs/heads/main/boston.csv'
data = pd.read_csv(url)
X = data.drop('target', axis=1)
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

ols = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=10).fit(X_train, y_train)

plt.figure(figsize=(8,5))
plt.bar(range(len(ols.coef_)), ols.coef_, label='OLS')
plt.bar(range(len(ridge.coef_)), ridge.coef_, alpha=0.6, label='Ridge (λ=10)')
plt.legend()
plt.title('Сравнение коэффициентов OLS и Ridge')
plt.show()


# %%

sigma = np.linspace(0.1, 10, 100)
for lam in [0.1, 1, 10, 100]:
    plt.plot(sigma, sigma**2/(sigma**2+lam), label=f'λ={lam}')
plt.title('Подавление факторов σᵢ² / (σᵢ² + λ)')
plt.xlabel('σᵢ')
plt.ylabel('Фактор подавления')
plt.legend()
plt.show()



