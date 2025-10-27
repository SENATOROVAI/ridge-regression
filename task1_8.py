def analyze_alpha_performance():
    print("=== АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ ПРИ РАЗНЫХ α ===\n")
    
    # Анализируем несколько ключевых значений alpha
    key_alphas = [1e-7, 1e-3, 1e-1, 1e1, 1e3]
    key_indices = [0, 2, 4, 6, 8]
    
    for idx, alpha_idx in enumerate(key_indices):
        model = models[alpha_idx]
        alpha_val = alphas[alpha_idx]
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        coef_norm = np.linalg.norm(model.coef_)
        
        print(f"α = {alpha_val:8.1e}:")
        print(f"  Train R²: {train_score:.4f}")
        print(f"  Test R²:  {test_score:.4f}")
        print(f"  Norm:     {coef_norm:.4f}")
        
        if alpha_val <= 1e-5:
            print("  → Почти как OLS, риск переобучения")
        elif alpha_val <= 1e-1:
            print("  → Хороший баланс")
        elif alpha_val <= 1e2:
            print("  → Умеренная регуляризация")
        else:
            print("  → Сильная регуляризация, риск недобучения")
        print()

analyze_alpha_performance()
