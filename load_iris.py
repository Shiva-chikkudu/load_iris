from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
iris = load_iris()
np.random(90)
compound1 = np.random.rand(56*3)
compound2 = np.random.rand(34*4)

# Features (X) and target (y)
X = iris.data        # shape (150, 4)
y = iris.target      # shape (150,)

print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("First 5 samples:\n", X[:5])
print("First 5 labels:", y[:5])

plt.plot(X, y, label="iris_data")
plt.title("IRIS DATA")
plt.legend()
plt.xlabel("x")
plt.ylabel("Y")
plt.show()
for i, feature_name in enumerate(iris.feature_names):
    plt.scatter(X[:, i], y, label=feature_name, alpha=0.6)
plt.title("IRIS DATA")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.show()