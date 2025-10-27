from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import numpy as np

X,y = load_diabetes(return_X_y=True)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

from sklearn.linear_model import Ridge
reg = Ridge(alpha=0.1,solver='cholesky')
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)
r2_score(y_test,y_pred)
