selected_alphas = [1e-7, 1e-3, 1e-1, 1e1, 1e3]
selected_models = [models[0], models[3], models[5], models[7], models[9]]

plt.figure(figsize=(15, 10))
plt.plot(X, y, color='g', linewidth=3, label='True function')
plt.scatter(x_train, y_train, s=80, label='Train data', zorder=5)
plt.scatter(x_test, y_test, s=80, label='Test data', c='r', zorder=5)

colors = ['red', 'blue', 'orange', 'purple', 'brown']
for i, (model, alpha) in enumerate(zip(selected_models, selected_alphas)):
    plt.plot(x_t, model.predict(X_t), '--', 
             color=colors[i], linewidth=2, 
             label=f'α = {alpha:.1e}')

plt.ylim(-50, 100)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Сравнение предсказаний Ridge-регрессии при разных α')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
