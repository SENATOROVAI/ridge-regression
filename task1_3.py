alphas_extended = np.logspace(-7, 5, 30)
train_scores = []
test_scores = []

for alpha in alphas_extended:
    ridge = Ridge(alpha=alpha).fit(X_train, y_train)
    train_scores.append(ridge.score(X_train, y_train))
    test_scores.append(ridge.score(X_test, y_test))

plt.figure(figsize=(12, 6))
plt.plot(alphas_extended, train_scores, 'b-', label='Train R²', linewidth=2)
plt.plot(alphas_extended, test_scores, 'r-', label='Test R²', linewidth=2)
plt.xscale('log')
plt.xlabel('α (логарифмическая шкала)')
plt.ylabel('R² Score')
plt.title('Bias-Variance Tradeoff в Ridge-регрессии')
plt.legend()
plt.grid(True, alpha=0.3)

# Показываем оптимальную точку
optimal_idx = np.argmax(test_scores)
plt.axvline(x=alphas_extended[optimal_idx], color='green', linestyle='--', 
           label=f'Optimal α = {alphas_extended[optimal_idx]:.2e}')
plt.legend()
plt.show()
