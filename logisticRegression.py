import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('daily_food_delivery_orders.csv')
df['target'] = df['order_status'].apply(lambda x: 1 if x == 'Delivered' else 0)

X = df[['delivery_distance_km', 'delivery_partner_rating']].values
y = df['target'].values

# Normalize & Add Intercept
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X = np.hstack([np.ones((X.shape[0], 1)), X])

split_idx = int(0.8 * len(y))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5 
    return (-1/m) * (y @ np.log(h + epsilon) + (1 - y) @ np.log(1 - h + epsilon))

def gradient_descent(X, y, theta, lr, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta -= lr * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

theta = np.zeros(X_train.shape[1])
theta_final, costs = gradient_descent(X_train, y_train, theta, lr=0.1, iterations=1000)

predictions = sigmoid(X_test @ theta_final) >= 0.5
print(f"Model Accuracy: {np.mean(predictions == y_test):.2f}")