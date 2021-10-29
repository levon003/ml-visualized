A = np.matmul(X_train.T, X_train)
A.shape

A_inv = np.linalg.pinv(A)

w = A_inv.dot(X_train.T).dot(y_train)
w.shape

