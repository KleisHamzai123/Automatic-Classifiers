from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

# Load MNIST-Dataset
digits = datasets.load_digits()

# Split Data into Training and Test sets
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False)

# Create KNN classifier with Euclidean distance
knn_euclidean = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

# Train model with training set
knn_euclidean.fit(X_train, y_train)

# Prediction on the test set
predicted_euclidean = knn_euclidean.predict(X_test)

# Show classification report
print(f"Classification report for KNN (Euclidean) classifier:\n"
      f"{metrics.classification_report(y_test, predicted_euclidean)}\n")

# Accuracy of the model
accuracy_euclidean = accuracy_score(y_test, predicted_euclidean)
print(f"The accuracy of the model using Euclidean distance is {accuracy_euclidean*100:.2f}%")

# Confusion matrix
cm_euclidean = confusion_matrix(y_test, predicted_euclidean)

# Show confusion matrix
plt.figure(figsize=(10, 10))
plt.imshow(cm_euclidean, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix (Euclidean)')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10), rotation=45)
plt.yticks(tick_marks, range(10))
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Show some misclassifications for Euclidean
misclassified_idx_euclidean = np.where(predicted_euclidean != y_test)[0]
sample_idx_euclidean = misclassified_idx_euclidean[:10]
fig, ax = plt.subplots(2, 5, figsize=(10, 5))
for i, idx in enumerate(sample_idx_euclidean):
    ax[i//5, i%5].imshow(X_test[idx].reshape(8, 8), cmap='gray')
    ax[i//5, i%5].set_title(f'True: {y_test[idx]}, Predict: {predicted_euclidean[idx]}')
plt.show()


# KNN classifier with Minkowski distance (p=3 same as previous)
knn_minkowski = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=3)

# Train model with training set
knn_minkowski.fit(X_train, y_train)

# Prediction on test set
predicted_minkowski = knn_minkowski.predict(X_test)

# Show classification report
print(f"Classification report for KNN (Minkowski p=3) classifier:\n"
      f"{metrics.classification_report(y_test, predicted_minkowski)}\n")

# Accuracy of the model
accuracy_minkowski = accuracy_score(y_test, predicted_minkowski)
print(f"The accuracy of the model using Minkowski distance (p=3) is {accuracy_minkowski*100:.2f}%")

# Confusion matrix
cm_minkowski = confusion_matrix(y_test, predicted_minkowski)

# Show confusion matrix
plt.figure(figsize=(10, 10))
plt.imshow(cm_minkowski, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix (Minkowski p=3)')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10), rotation=45)
plt.yticks(tick_marks, range(10))
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Show some misclassifications for Minkowski
misclassified_idx_minkowski = np.where(predicted_minkowski != y_test)[0]
sample_idx_minkowski = misclassified_idx_minkowski[:10]
fig, ax = plt.subplots(2, 5, figsize=(10, 5))
for i, idx in enumerate(sample_idx_minkowski):
    ax[i//5, i%5].imshow(X_test[idx].reshape(8, 8), cmap='gray')
    ax[i//5, i%5].set_title(f'True: {y_test[idx]}, Predict: {predicted_minkowski[idx]}')
plt.show()
