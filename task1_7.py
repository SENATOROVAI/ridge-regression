# Добавляем больше шума к обучающим данным
y_train_noisy = f(x_train) + 20 * np.random.randn(10)

plt.figure(figsize=(15, 6))

# Без регуляризации
plt.subplot(1, 2, 1)
lr_noisy = LinearRegression().fit(X_train, y_train_noisy)
plt.plot(X, y, color='g', label='True function')
plt.scatter(x_train, y_train_noisy, label='Noisy train data')
plt.plot(x_t, lr_noisy.predict(X_t), 'r--', label='OLS')
plt.ylim(-80, 120)
plt.title('OLS на зашумленных данных')
plt.legend()

# С регуляризацией
plt.subplot(1, 2, 2)
ridge_noisy = Ridge(alpha=10).fit(X_train, y_train_noisy)
plt.plot(X, y, color='g', label='True function')
plt.scatter(x_train, y_train_noisy, label='Noisy train data')
plt.plot(x_t, ridge_noisy.predict(X_t), 'b--', label='Ridge (α=10)')
plt.ylim(-80, 120)
plt.title('Ridge на зашумленных данных')
plt.legend()

plt.tight_layout()
plt.show()

print(f"OLS test R2: {lr_noisy.score(X_test, y_test):.4f}")
print(f"Ridge test R2: {ridge_noisy.score(X_test, y_test):.4f}")
