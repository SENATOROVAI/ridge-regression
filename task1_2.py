from sklearn.model_selection import cross_val_score

best_alpha = None
best_score = -float('inf')

for alpha in np.logspace(-5, 3, 20):
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')
    mean_score = np.mean(scores)
    
    if mean_score > best_score:
        best_score = mean_score
        best_alpha = alpha

print(f"Оптимальный α: {best_alpha:.6f}")
print(f"Лучший R2 score: {best_score:.4f}")

# Обучаем модель с оптимальным alpha
optimal_ridge = Ridge(alpha=best_alpha).fit(X_train, y_train)
print(f"Test R2 с оптимальным α: {optimal_ridge.score(X_test, y_test):.4f}")
