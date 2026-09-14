from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Training data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 5, 7, 9, 11])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predictions
predictions = model.predict(X)

# Display predictions
for x, prediction in zip(X, predictions):
    print(f"Input: {x[0]} -> Prediction: {prediction:.2f}")

# Plot actual data
plt.scatter(X, y, label="Training Data")

# Plot learned line
plt.plot(X, predictions, label="AI Model")

plt.xlabel("Input")
plt.ylabel("Output")
plt.title("AI Number Predictor")

plt.legend()
plt.show()