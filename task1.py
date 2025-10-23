from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X3, y3 = make_regression(n_samples=14, n_features=12, noise=2, random_state=0)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, random_state=0)

sk_ridge_regression = Ridge()
sk_ridge_regression.fit(X3_train, y3_train)

sk_ridge_pred_train_res = sk_ridge_regression.predict(X3_train)

sk_ridge_r2 = r2_score(y3_test, sk_ridge_pred_res)
sk_ridge_train_r2 = r2_score(y3_train, sk_ridge_pred_train_res)

sk_ridge_mse = mean_squared_error(y3_test, sk_ridge_pred_res)
sk_ridge_train_mse = mean_squared_error(y3_train, sk_ridge_pred_train_res)

sk_ridge_mape = mean_absolute_percentage_error(y3_test, sk_ridge_pred_res)
sk_ridge_train_mape = mean_absolute_percentage_error(y3_train, sk_ridge_pred_train_res)

print(f'Ridge R2 score: {sk_ridge_r2}')
print(f'Ridge train R2 score: {sk_ridge_train_r2}', '\n')

print(f'Ridge MSE: {sk_ridge_mse}')
print(f'Ridge train MSE: {sk_ridge_train_mse}', '\n')

print(f'Ridge MAPE: {sk_ridge_mape}')
print(f'Ridge train MAPE: {sk_ridge_train_mape}', '\n')



