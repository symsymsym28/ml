import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Sample Data (or you can load your own CSV)
np.random.seed(42)
X_data = 2 * np.random.rand(100, 1)
y_data = 4 + 3 * X_data + np.random.randn(100, 1) # y = 4 + 3x + noise

# 2. Add Intercept Term (column of ones) to X
X = np.hstack([np.ones((X_data.shape[0], 1)), X_data])
y = y_data

# 3. Define the Cost Function (Mean Squared Error)
def compute_cost_linear(X, y, theta):
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# 4. Define Gradient Descent
def gradient_descent_linear(X, y, theta, lr, iterations):
    m = len(y)
    cost_history = []
    
    for _ in range(iterations):
        predictions = X @ theta
        gradient = (1 / m) * (X.T @ (predictions - y))
        theta -= lr * gradient
        cost_history.append(compute_cost_linear(X, y, theta))
        
    return theta, cost_history

# 5. Initialize Parameters and Train
theta = np.zeros((X.shape[1], 1))
learning_rate = 0.1
iterations = 1000

theta_final, costs = gradient_descent_linear(X, y, theta, learning_rate, iterations)

print(f"Learned Parameters: Intercept = {theta_final[0][0]:.2f}, Slope = {theta_final[1][0]:.2f}")

# 6. Visualize the Results
plt.figure(figsize=(12, 5))

# Plot Cost History
plt.subplot(1, 2, 1)
plt.plot(costs)
plt.title('Cost vs Iterations')
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')

# Plot Regression Line
plt.subplot(1, 2, 2)
plt.scatter(X_data, y_data, color='blue', label='Actual Data', alpha=0.6)
plt.plot(X_data, X @ theta_final, color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()