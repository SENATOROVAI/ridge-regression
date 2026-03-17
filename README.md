#### https://SenatorovAI.com

# L2 Regularization (Ridge Regression) — Data Science & Machine Learning

[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Website](https://img.shields.io/badge/website-live-blue.svg)](https://senatorovai.github.io/L2-regularization-ridge-regression-course/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18818738.svg)](https://doi.org/10.5281/zenodo.18821472)
[![Code Style](https://img.shields.io/badge/code%20style-black-black)]()
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)]()

**Keywords:** L2 regularization, Ridge regression, Tikhonov regularization, bias-variance tradeoff, overfitting, linear regression, machine learning, data science, regularization parameter lambda

---

## 📌 Overview

This repository explains **L2 regularization (Ridge Regression)** from scratch:

* Mathematical derivation
* Geometric intuition
* Bias–variance tradeoff
* Closed-form solution
* Optimization perspective
* Numerical stability
* Practical implementation in Python


---

# 1️⃣ Why Do We Need L2 Regularization?

In Linear Regression we minimize:

$$
J(w) = |Xw - y|^2
$$

The solution is:

$$
w^* = (X^T X)^{-1} X^T y
$$

### Problem

If:

* Features are highly correlated (multicollinearity)
* Number of features is large
* $$X^T X$$ is ill-conditioned or nearly singular

Then:

* We get unstable coefficients
* Large variance
* Overfitting

---

# 2️⃣ What Is L2 Regularization?

L2 regularization adds a penalty on the squared magnitude of weights:

$$
J(w) = |Xw - y|^2 + \lambda |w|^2
$$

Where:

$$
|w|^2 = w^T w
$$

and:

* $$\lambda \ge 0$$ — regularization parameter
* Controls model complexity

---

# 3️⃣ Ridge Regression Closed-Form Solution

To minimize:

$$
J(w) = (Xw - y)^T (Xw - y) + \lambda w^T w
$$

Take gradient:

$$
\nabla J(w) = 2X^T X w - 2X^T y + 2\lambda w
$$

Set to zero:

$$
X^T X w + \lambda w = X^T y
$$

Factor:

$$
(X^T X + \lambda I) w = X^T y
$$

Final solution:

$$
w^* = (X^T X + \lambda I)^{-1} X^T y
$$

---

# 4️⃣ Why Does It Work?

### Key Insight

Adding $$\lambda I$$:

* Shifts eigenvalues of $$X^T X$$
* Improves conditioning
* Makes matrix invertible
* Shrinks coefficients toward zero

If eigenvalues of $$X^T X$$ are $$\sigma_i^2$$

Then eigenvalues become:

$$
\sigma_i^2 + \lambda
$$

This prevents division by very small numbers.

---

# 5️⃣ Geometric Interpretation

OLS solution:

* Minimizes error
* No constraint on weight magnitude

Ridge solution:

Minimize:

$$
|Xw - y|^2
$$

Subject to:

$$
|w|^2 \le c
$$

Equivalent constrained optimization problem.

Geometrically:

* Error contours = ellipses
* Constraint = circle
* Solution = tangency point

---

# 6️⃣ Bias–Variance Tradeoff

Expected prediction error:

$$
\mathbb{E}[(y - \hat{f}(x))^2] =
\text{Bias}^2 + \text{Variance} + \sigma^2
$$

As $$\lambda$$ increases:

* Bias ↑
* Variance ↓
* Model complexity ↓

Proper $$\lambda$$ balances bias and variance.

---

# 7️⃣ Ridge vs Ordinary Least Squares

| Property                  | OLS | Ridge |
| ------------------------- | --- | ----- |
| Closed-form solution      | ✅   | ✅     |
| Handles multicollinearity | ❌   | ✅     |
| Shrinks coefficients      | ❌   | ✅     |
| Feature selection         | ❌   | ❌     |
| Works when p > n          | ❌   | ✅     |

---

# 8️⃣ Optimization Perspective

Ridge regression minimizes:

$$
J(w) = \frac{1}{n}|Xw - y|^2 + \lambda |w|^2
$$

Gradient:

$$
\nabla J(w) = \frac{2}{n}X^T(Xw - y) + 2\lambda w
$$

Can be solved using:

* Gradient Descent
* Newton's Method
* Conjugate Gradient
* L-BFGS

---

# 9️⃣ Connection to Tikhonov Regularization

Ridge regression is equivalent to:

**Tikhonov regularization**

General form:

$$
|Xw - y|^2 + \lambda |Lw|^2
$$

Ridge is special case:

$$
L = I
$$

---

# 🔟 Choosing λ (Regularization Strength)

Common methods:

* Cross-validation
* Grid search
* Analytical shrinkage paths
* Generalized Cross Validation (GCV)

As:

* $$\lambda \to 0$$ → Ridge → OLS
* $$\lambda \to \infty$$ → $$w \to 0$$

---

# 1️⃣1️⃣ Implementation in Python

```python
import numpy as np
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X, y)

print(model.coef_)
```

---

# 1️⃣2️⃣ Ridge in High Dimensions (p > n)

When:

$$
p > n
$$

OLS is not uniquely defined.

Ridge solution still exists because:

$$
X^T X + \lambda I
$$

is always invertible for $$\lambda > 0$$.

---

# 1️⃣3️⃣ Spectral View (SVD Perspective)

If:

$$
X = U \Sigma V^T
$$

Then Ridge solution becomes:

$$
w^* = V (\Sigma^T \Sigma + \lambda I)^{-1} \Sigma^T U^T y
$$

Shrinkage happens along singular directions:

$$
\frac{\sigma_i}{\sigma_i^2 + \lambda}
$$

Small singular values → stronger shrinkage.

---

# 🔎 SEO Keywords (GitHub & Google)

ridge regression tutorial, L2 regularization explained, ridge regression math derivation, bias variance tradeoff, multicollinearity solution, tikhonov regularization, machine learning regularization, linear regression with penalty, shrinkage methods, data science mathematics

---

# 🎯 Target Audience

* Data Science students
* Machine Learning engineers
* Researchers studying generalization
* Anyone learning linear models and regularization

---

# 📚 Related Topics

* L1 Regularization (Lasso)
* Elastic Net
* Bias–Variance Tradeoff
* Cross-Validation
* Singular Value Decomposition (SVD)
* Condition Number

---

⭐ If this repository helped you understand L2 regularization and Ridge regression, consider giving it a star.
