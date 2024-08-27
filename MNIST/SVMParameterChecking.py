from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# Load MNIST-Datenset
digits = datasets.load_digits()


# Split Data into Trainingsset and Testset
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False)

# Parameter value to be checked
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

# Create GridSearchCV
search = GridSearchCV(svm.SVC(), param_grid, cv=5)
search.fit(X_train, y_train)

# Print out the best Parameter
print(f"Die besten Parameter sind {search.best_params_}")
# Create SVM-Classification using the best Parameter
best_params = search.best_params_
classifier = svm.SVC(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params.get('gamma'))

# Train model with Trainingsset
classifier.fit(X_train, y_train)

# Prediction on Testset
predicted = classifier.predict(X_test)


# Accuracy of model
accuracy = accuracy_score(y_test, predicted)
print(f"Die Genauigkeit des Modells ist {accuracy*100:.2f}%")

# Ausgabe der Klassifikationsberichte
print(f"Classification report for classifier {classifier}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")