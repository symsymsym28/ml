import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('main.csv')
X_data = df.iloc[:, 0].values.reshape(-1, 1)
y_data = df.iloc[:, 1].values.reshape(-1, 1)

X = np.hstack([np.ones((X_data.shape[0], 1)), X_data])
y = y_data

def compute_cost_linear(X, y, theta):
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent_linear(X, y, theta, lr, iterations):
    m = len(y)
    cost_history = []
    
    for _ in range(iterations):
        predictions = X @ theta
        gradient = (1 / m) * (X.T @ (predictions - y))
        theta -= lr * gradient
        cost_history.append(compute_cost_linear(X, y, theta))
        
    return theta, cost_history

theta = np.zeros((X.shape[1], 1))
learning_rate = 0.01
iterations = 1000

theta_final, costs = gradient_descent_linear(X, y, theta, learning_rate, iterations)

print(f"Learned Parameters: Intercept = {theta_final[0][0]:.2f}, Slope = {theta_final[1][0]:.2f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(costs)
plt.title('Cost vs Iterations')
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')

plt.subplot(1, 2, 2)
plt.scatter(X_data, y_data, color='blue', alpha=0.6)
plt.plot(X_data, X @ theta_final, color='red', linewidth=2)
plt.title('Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('y')

plt.tight_layout()
plt.show()
