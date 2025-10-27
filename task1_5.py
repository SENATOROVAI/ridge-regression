norms = []
for model in models[:-1]:  # Все Ridge модели
    norms.append(np.linalg.norm(model.coef_))

# Норма для LinearRegression
lr_norm = np.linalg.norm(models[-1].coef_)

plt.figure(figsize=(10, 6))
plt.plot(alphas, norms, 'bo-', linewidth=2, markersize=6)
plt.axhline(y=lr_norm, color='red', linestyle='--', 
           label=f'OLS норма: {lr_norm:.2f}')
plt.xscale('log')
plt.xlabel('α')
plt.ylabel('L2-норма коэффициентов')
plt.title('Влияние α на величину коэффициентов')
plt.legend()
plt.grid(True, alpha=0.3)

print(f"L2-норма коэффициентов OLS: {lr_norm:.4f}")
print(f"L2-норма при α={alphas[0]}: {norms[0]:.4f}")
print(f"L2-норма при α={alphas[-1]}: {norms[-1]:.4f}")
print(f"Сжатие: {((lr_norm - norms[-1])/lr_norm * 100):.1f}%")
