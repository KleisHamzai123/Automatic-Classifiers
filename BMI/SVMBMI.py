import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('height_weight_bmi_samples3.csv')

# Extract features (X) and target (y)
X = data[['Height (cm)', 'Weight (kg)']]
y = data['BMI Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create SVM model with margin 5
svm_model = SVC(kernel='linear', C=0.1, random_state=1)

# Train the model on the training data
svm_model.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = svm_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Define the meshgrid range
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict the labels for each point in meshgrid
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Get support vectors and coefficients
support_vectors = svm_model.support_vectors_
coefficients = svm_model.coef_[0]

# Calculate the margin for each support vector
margins = np.abs(svm_model.decision_function(support_vectors)) / np.linalg.norm(coefficients)

# Find the maximum margin
max_margin = np.min(margins)

print("Maximum Margin:", max_margin)

# Plot the decision boundaries
plt.contourf(xx, yy, Z, alpha=0.4)

# Plot support vectors
plt.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k', linewidths=2, label='Support Vectors')

# Plot data points
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=20, edgecolor='k', label='Data Points')

plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('SVM Decision Boundaries with Support Vectors')
plt.legend()
plt.show()