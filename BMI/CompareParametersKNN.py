import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Load the created data
data = pd.read_csv('height_weight_bmi_samples3.csv')

# Extract features (X) and target (y)
X = data[['Height (cm)', 'Weight (kg)']]
y = data['BMI Class']

# Initialize the KNN classifiers for k = 3 and k = 5
k_values = [3, 5]
classifiers = [KNeighborsClassifier(n_neighbors=k) for k in k_values]

# Train the KNN classifiers to learn relationships between data
for classifier in classifiers:
    classifier.fit(X, y)

# Create a graph with zones to visualize the decision boundaries
h = 0.8
x_min, x_max = X['Height (cm)'].min() - 1, X['Height (cm)'].max() + 1
y_min, y_max = X['Weight (kg)'].min() - 1, X['Weight (kg)'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict the class labels for each data point into one-dimensional array
Zs = []
for classifier in classifiers:
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    Zs.append(Z)

# Create color maps with stronger colors
cmap_light = ListedColormap(['#FFCCCC', '#CCFFCC', '#CCCCFF', '#FFD699'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFA500'])

# Plot the decision boundaries and data points for each k value
plt.figure(figsize=(16, 6))
for i, k in enumerate(k_values):
    plt.subplot(1, 2, i+1)
    plt.pcolormesh(xx, yy, Zs[i], cmap=cmap_light)
    plt.scatter(X['Height (cm)'], X['Weight (kg)'], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.title(f'BMI Class Decision Boundaries (k={k})')

plt.tight_layout()
plt.show()