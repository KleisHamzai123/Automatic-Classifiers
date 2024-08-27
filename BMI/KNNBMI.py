import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score



class EuclideandistanceKNN:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances from x to all examples in the training set
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.n_neighbors]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_params(self, deep=True):
        return {"n_neighbors": self.n_neighbors}

    def set_params(self, **params):
        if "n_neighbors" in params:
            self.n_neighbors = params["n_neighbors"]
        return self

# Load the created data
data = pd.read_csv('height_weight_bmi_samples3.csv')

# Extract features (X) and target (y)
X = data[['Height (cm)', 'Weight (kg)']]
y = data['BMI Class']

# Define values of k for the custom KNN
k_values = [3, 5, 7, 9]

# Initialize the custom KNN classifiers for k = 3,5,7,9
classifiers = [EuclideandistanceKNN(n_neighbors=k) for k in k_values]

# Perform cross-validation to find optimal k and evaluate performance
cv_scores = []
for classifier in classifiers:
    scores = cross_val_score(classifier, X.values, y.values, cv=3, scoring='accuracy')
    cv_scores.append(scores.mean())

# Print cross-validation scores for each value of k
for k, score in zip(k_values, cv_scores):
    print(f"Cross-validation accuracy for k={k}: {score}")
