from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1000)
ridge.fit(X, y)
print(ridge.coef_)
