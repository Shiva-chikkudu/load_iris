from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import seaborn as sns
iris = load_iris()
X = iris.data
Y = iris.target
X_train,Y_train,X_test,Y_test = train_test_split(X,Y, test_size = 0.4, random_state=50)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
precision = precision_score(Y_test, predictions, average='weighted')
recall = recall_score(Y_test, predictions, average='weighted')
plt.figure(figsize=(5, 8))
plt.plot([12,15, 19, 25], [accuracy, precision, recall], marker='o')
plt.title("Model Performance Metrics")
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.xticks([12,15, 19, 25], ["Accuracy", "Precision", "Recall"])
plt.show()
