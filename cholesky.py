import numpy as np

def ridge_cholesky(X, y, lambda_val):
    n, p = X.shape
    # Вычисляем матрицу системы
    A = X.T @ X + lambda_val * np.eye(p)
    
    # Вычисляем правую часть
    b = X.T @ y
    
    # Разложение Холецкого
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        print("Матрица не положительно определена")
        return None
    
    # Решение системы: сначала Ly = b
    y_sol = np.linalg.solve(L, b)
    
    # Затем L^T beta = y
    beta = np.linalg.solve(L.T, y_sol)
    
    return beta

# Тестирование
X = np.random.randn(100, 5)
y = np.random.randn(100)
beta = ridge_cholesky(X, y, 1.0)
print("Коэффициенты:", beta)
