# %% [markdown]
# # Регуляризация

# %%
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # Регрессия

# %% [markdown]
# Допустим, у нас есть исходная известная зависимость 3-го порядка:
# 
# $$f(x) = 0.6 - 13.2x - 5.3 x^{2} - 4.17x^{3}.$$
# 
# Реализуем ее в виде python-функции и построим график.

# %%
import numpy as np

# %%
def f(x):
    return 0.4 + 2 * x + 0.8 * x ** 2 + 0.5 * x ** 3

# %%
X = np.linspace(-5, 5, 100)
y = f(X)

# %%
import matplotlib.pyplot as plt

plt.xlabel('x')
plt.ylabel('f(x)')

plt.plot(X, y, color='g');

# %% [markdown]
# Теперь сгенерируем датасет из десяти случайных точек, подчиняющихся этой зависимости, с добавлением шума и нанесем на график.

# %%
np.random.seed(18)
x_train = np.random.uniform(-5, 5, 10)
y_train = f(x_train) + 10 * np.random.randn(10)

# %%
plt.figure(figsize=(10, 6))
plt.xlabel('x')
plt.ylabel('f(x)')

plt.plot(X, y, color='g')
plt.scatter(x_train, y_train, label='train data');

# %%
np.random.seed(8)
x_test = np.random.uniform(-5, 5, 10)
x_test = np.sort(x_test)
y_test = f(x_test)

X_test = x_test.reshape(-1, 1)
X_train = x_train.reshape(-1, 1)

X_train.shape

# %%
plt.figure(figsize=(10, 6))
plt.xlabel('x')
plt.ylabel('f(x)')

plt.plot(X, y, color='g')
plt.scatter(x_train, y_train, label='train data');
plt.scatter(x_test, y_test, label='test data', c='r')
plt.legend();

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

pipe = make_pipeline(PolynomialFeatures(11, include_bias=False),
                     StandardScaler())

pipe.fit(X_train)

# %%
X_train = pipe.transform(X_train)
X_test = pipe.transform(X_test)

X_train

# %% [markdown]
# ## Ridge (L2 регуляризация)

# %%
alphas = list(np.logspace(-7, 4, 10))
models = []
coefs = []

# %%
alphas 

# %%
from sklearn.linear_model import Ridge, LinearRegression

for alpha in alphas:
    m_r = Ridge(alpha=alpha).fit(X_train, y_train)
    models.append(m_r)
    coefs.append(m_r.coef_)

models.append(LinearRegression().fit(X_train, y_train))

# %%
len(models)

# %%
len(coefs)

# %%
coefs[0].shape

# %% [markdown]
# ### Визуализация предсказаний

# %%
plt.figure(figsize=(15, 10))
plt.xlabel('x')
plt.ylabel('f(x)')

plt.plot(X, y, color='g')
plt.scatter(x_train, y_train, label='train data');
plt.scatter(x_test, y_test, label='test data', c='r')
plt.legend()

x_t = np.sort(np.random.uniform(-5, 5, 100)).reshape(-1, 1)
X_t = pipe.transform(x_t)

for m in models:
    plt.plot(x_t, m.predict(X_t), '--')

plt.plot(x_t, models[-1].predict(X_t), 'b-')
plt.ylim(-50, 100);

# %%
import pandas as pd

scores = pd.DataFrame()
scores = scores.append(
        {
            'alpha': 'no',
            'train_r2': models[-1].score(X_train, y_train),
            'test_r2': models[-1].score(X_test, y_test)
        }, ignore_index=True
    )

for i, m in enumerate(models[:-1]):
    alpha = alphas[i]
    scores = scores.append(
        {
            'alpha': alpha,
            'train_r2': m.score(X_train, y_train),
            'test_r2': m.score(X_test, y_test)
        }, ignore_index=True
    )


scores

# %% [markdown]
# ### Визуализация весов

# %%
coefs

# %%
np.vstack(coefs).T

# %%
plt.figure(figsize=(15, 10))

for i in np.vstack(coefs).T:
    plt.plot(alphas, i, linewidth=3)

plt.grid()
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('weight');


