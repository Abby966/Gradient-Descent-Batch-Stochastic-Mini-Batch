"""Abegail Chanyalew -----UGR/0964/15 
Henock Getahun --------UGR/5136/15
Kalkidan Gashaw -------UGR/1863/15
Andinet Siyoum --------UGR/9364/15
"""

import numpy as np
import matplotlib.pyplot as plt
# Create sample data
np.random.seed(42)

X = np.linspace(0, 10, 100)   # 100 values between 0 and 10
y = 3 * X + 2 + np.random.randn(100) * 2
X = (X - X.mean()) / X.std()
w = 0.0
b = 0.0
learning_rate = 0.01
def compute_loss(X, y, w, b):
    y_pred = w * X + b
    return np.mean((y - y_pred) ** 2)
def batch_gradient_descent(X, y, w, b, lr, epochs):
    losses = []

    for _ in range(epochs):
        y_pred = w * X + b

        dw = -2 * np.mean(X * (y - y_pred))
        db = -2 * np.mean(y - y_pred)

        w -= lr * dw
        b -= lr * db

        losses.append(compute_loss(X, y, w, b))

    return w, b, losses
def stochastic_gradient_descent(X, y, w, b, lr, epochs):
    losses = []

    for _ in range(epochs):
        for i in range(len(X)):
            y_pred = w * X[i] + b

            dw = -2 * X[i] * (y[i] - y_pred)
            db = -2 * (y[i] - y_pred)

            w -= lr * dw
            b -= lr * db

        losses.append(compute_loss(X, y, w, b))

    return w, b, losses
def mini_batch_gradient_descent(X, y, w, b, lr, epochs, batch_size=10):
    losses = []

    for _ in range(epochs):
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, len(X), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            y_pred = w * X_batch + b

            dw = -2 * np.mean(X_batch * (y_batch - y_pred))
            db = -2 * np.mean(y_batch - y_pred)

            w -= lr * dw
            b -= lr * db

        losses.append(compute_loss(X, y, w, b))

    return w, b, losses
epochs = 50

w_b, b_b, loss_b = batch_gradient_descent(X, y, 0, 0, learning_rate, epochs)
w_s, b_s, loss_s = stochastic_gradient_descent(X, y, 0, 0, learning_rate, epochs)
w_m, b_m, loss_m = mini_batch_gradient_descent(X, y, 0, 0, learning_rate, epochs)
plt.plot(loss_b, label="Batch GD")
plt.plot(loss_s, label="Stochastic GD")
plt.plot(loss_m, label="Mini-Batch GD")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
