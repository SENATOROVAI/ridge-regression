degrees = [3, 5, 7, 11]
plt.figure(figsize=(15, 10))

for i, degree in enumerate(degrees, 1):
    plt.subplot(2, 2, i)
    
    # Создаем pipeline для каждой степени
    pipe_deg = make_pipeline(PolynomialFeatures(degree, include_bias=False),
                           StandardScaler())
    X_train_deg = pipe_deg.fit_transform(X_train.reshape(-1, 1))
    X_t_deg = pipe_deg.transform(x_t)
    
    # Обучаем без регуляризации
    lr = LinearRegression().fit(X_train_deg, y_train)
    
    plt.plot(X, y, color='g', label='True function')
    plt.scatter(x_train, y_train, label='Train data')
    plt.plot(x_t, lr.predict(X_t_deg), 'r--', label=f'Degree {degree}')
    plt.ylim(-50, 100)
    plt.legend()
    plt.title(f'Polynomial Degree {degree}\nTrain R2: {lr.score(X_train_deg, y_train):.3f}')

plt.tight_layout()
plt.show()
