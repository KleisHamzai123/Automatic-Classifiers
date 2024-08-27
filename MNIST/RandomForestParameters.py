from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

# Load MNIST Dataset
digits = datasets.load_digits()

# Split Data into Training and Test sets
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, random_state=42)

# Hyperparameter Tuning
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                   param_distributions=param_dist,
                                   n_iter=75, cv=5,
                                   scoring='accuracy',
                                   random_state=42)

# Split X_train and y_train for hyperparameter tuning
X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Fit the random search model
random_search.fit(X_train_cv, y_train_cv)

# Print the best hyperparameters found by random search
print("\nBest hyperparameters:", random_search.best_params_)

# Use the best model to make predictions on the validation set
best_rf_model = random_search.best_estimator_
y_pred_val = best_rf_model.predict(X_val_cv)

# Model Evaluation

# Print classification report on the validation set
print("\nClassification Report (Validation Set):")
print(classification_report(y_val_cv, y_pred_val))

# Print confusion matrix on the validation set
print("\nConfusion Matrix (Validation Set):")
cm = confusion_matrix(y_val_cv, y_pred_val)
print(cm)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
ax = sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')  # fmt="d": display in integer values
ax.set_title('Confusion Matrix (Validation Set)')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.show()

# Test Set Evaluation

# Now evaluate the model on the test set
y_pred_test = best_rf_model.predict(X_test)

# Print classification report on the test set
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test))

# Print confusion matrix on the test set
print("\nConfusion Matrix (Test Set):")
cm_test = confusion_matrix(y_test, y_pred_test)
print(cm_test)

# Create a heatmap for the confusion matrix on the test set
plt.figure(figsize=(8, 6))
ax = sns.heatmap(cm_test, annot=True, fmt="d", cmap='Blues')
ax.set_title('Confusion Matrix (Test Set)')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.show()