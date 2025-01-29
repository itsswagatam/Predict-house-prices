import numpy as np
import matplotlib.pyplot as plt

# Generate data: House size (sq ft) and Price ($1000s)
np.random.seed(42)  # For reproducibility
X = 2.5 * np.random.rand(100, 1) + 0.5  # House size (normalized)
y = 50 + 30 * X + np.random.randn(100, 1) * 5  # Price in $1000s with noise

# Normalize data for better gradient descent performance
X_norm = (X - np.mean(X)) / np.std(X)

# Add bias term (x0 = 1)
X_b = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]  # Add x0 = 1 for bias term

# Define Linear Regression Model
def predict(X, theta):
    """Predict output using linear model."""
    return X @ theta

# Compute Cost Function
def compute_cost(X, y, theta):
    """Compute Mean Squared Error (MSE) cost."""
    m = len(y)
    predictions = predict(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Gradient Descent Algorithm
def gradient_descent(X, y, theta, learning_rate=0.1, iterations=1000):
    """Perform Gradient Descent to optimize theta."""
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        gradients = (1 / m) * X.T @ (predict(X, theta) - y)
        theta -= learning_rate * gradients
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        
        if i % 100 == 0:  # Print progress
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return theta, cost_history

# Initialize Theta
theta_init = np.random.randn(2, 1)  # Random initial theta

# Train Model
theta_opt, cost_history = gradient_descent(X_b, y, theta_init, learning_rate=0.1, iterations=1000)
print(f"Optimized Parameters: {theta_opt.flatten()}")

# Plot Cost Function Convergence
plt.plot(range(len(cost_history)), cost_history, 'b-')
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Cost Function Convergence")
plt.show()

# Plot Regression Line
plt.scatter(X, y, label="Training Data", alpha=0.7)
plt.plot(X, predict(X_b, theta_opt), color="red", label="Regression Line")
plt.xlabel("House Size (normalized)")
plt.ylabel("Price ($1000s)")
plt.legend()
plt.title("Linear Regression Fit")
plt.show()
