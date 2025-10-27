# Генерируем новый тестовый набор
np.random.seed(42)
x_new_test = np.random.uniform(-5, 5, 15)
y_new_test = f(x_new_test) + 8 * np.random.randn(15)

# Преобразуем новые данные
X_new_test = pipe.transform(x_new_test.reshape(-1, 1))

# Находим лучшую модель по исходному тесту
best_test_score = -float('inf')
best_model = None
best_alpha = None

for i, model in enumerate(models[:-1]):
    test_score = model.score(X_test, y_test)
    if test_score > best_test_score:
        best_test_score = test_score
        best_model = model
        best_alpha = alphas[i]

print(f"Лучшая модель: α = {best_alpha:.2e}")
print(f"Исходный test R²: {best_test_score:.4f}")

# Тестируем на новых данных
new_test_score = best_model.score(X_new_test, y_new_test)
print(f"Новый test R²: {new_test_score:.4f}")

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(X, y, color='g', linewidth=2, label='True function')
plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_new_test, y_new_test, label='New test data', c='orange', s=80)

# Предсказания лучшей модели
x_plot = np.linspace(-5, 5, 100).reshape(-1, 1)
X_plot = pipe.transform(x_plot)
plt.plot(x_plot, best_model.predict(X_plot), 'r--', 
         label=f'Best Ridge (α={best_alpha:.1e})')

plt.ylim(-50, 100)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'Валидация лучшей модели на новых данных\nNew Test R²: {new_test_score:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
