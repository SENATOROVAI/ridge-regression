def ridge_loss(y_true, y_pred, beta, lam):
    rss = sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true)))
    penalty = lam * sum(____ for b in beta)
    return rss + penalty
