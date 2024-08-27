from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# Load MNIST-Dataset
digits = datasets.load_digits()

# Split Data into Trainingsset and Testset
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False)

# Create SVM classifier
classifier = svm.SVC(gamma=0.005)

# Train modell with Trainingsset
classifier.fit(X_train, y_train)

# Prediction on Testset
predicted = classifier.predict(X_test)

# Show classification
print(f"Classification report for classifier {classifier}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")

# Accuracy of model
accuracy = accuracy_score(y_test, predicted)
print(f"Die Genauigkeit des Modells ist {accuracy*100:.2f}%")

# Create Confusionsmatrix
cm = confusion_matrix(y_test, predicted)

# Show Confusionsmatrix
plt.figure(figsize=(10,10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10), rotation=45)
plt.yticks(tick_marks, range(10))
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Show some  missclassifications
misclassified_idx = np.where(predicted != y_test)[0]
sample_idx = misclassified_idx[:10]
fig, ax = plt.subplots(2, 5, figsize=(10,5))
for i, idx in enumerate(sample_idx):
    ax[i//5, i%5].imshow(X_test[idx].reshape(8, 8), cmap='gray')
    ax[i//5, i%5].set_title(f'True: {y_test[idx]}, Predict: {predicted[idx]}')
plt.show()