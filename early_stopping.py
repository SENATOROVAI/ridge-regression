def fit(self, x, y, iter=20000, learning_rate=0.05, tol=1e-6, patience=100):
    x, y = x.copy(), y.copy()
    x = self.add_ones(x)
    thetas = np.zeros(x.shape[1])
    n = x.shape[0]
    loss_history = []
    best_loss = np.inf
    wait = 0

    for i in range(iter):
        loss = self.objective(x, y, thetas, n)
        loss_history.append(loss)

        if loss < best_loss - tol:
            best_loss = loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stopping at iteration {i}')
                break

        grad = self.gradient(x, y, thetas, n)
        thetas -= learning_rate * grad

    self.thetas = thetas
    self.loss_history = loss_history
