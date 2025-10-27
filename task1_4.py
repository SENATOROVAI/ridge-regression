plt.figure(figsize=(15, 8))

# Берем подмножество коэффициентов для наглядности
feature_indices = [0, 1, 5, 10]  # x, x², x⁶, x¹¹

for idx in feature_indices:
    coef_values = [model.coef_[idx] for model in models[:-1]]
    plt.plot(alphas, coef_values, linewidth=3, 
             label=f'x^{idx+1} coefficient', marker='o')

plt.xscale('log')
plt.xlabel('α')
plt.ylabel('Значение коэффициента')
plt.title('Сжатие коэффициентов при увеличении регуляризации')
plt.legend()
plt.grid(True, alpha=0.3)

# Показываем коэффициенты без регуляризации
lr_coefs = models[-1].coef_
print("Коэффициенты без регуляризации:")
for idx in feature_indices:
    print(f"x^{idx+1}: {lr_coefs[idx]:.4f}")
