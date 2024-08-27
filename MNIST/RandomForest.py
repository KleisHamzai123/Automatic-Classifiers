# Compare the usage of PCA and LDA regarding RF (done in Google Colab)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# Load the MNIST datasets
train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

# Feature Extraction ( y is label and X is features)
y_train = train.iloc[:, 0]  # Label is in the first column (index 0)
X_train = train.drop(train.columns[0], axis=1)  # Drop the first column (label)

# Extract features (X) and target variable (y) from test data
y_test = test.iloc[:, 0]
X_test = test.drop(test.columns[0], axis=1)

# Preprocess data
X_train = X_train / 255.0
X_test = X_test / 255.0

# PCA Preprocessing with Cross-Validation
# Explore different n_components values
n_components_list = range(1, min(X_train.shape[1], len(np.unique(y_train)) - 1) + 1)
cv_scores_pca = []

for n_components in n_components_list:
  # KFold cross-validation
  kf = KFold(n_splits=5, shuffle=True, random_state=42)

  fold_scores = []
  for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # PCA within the fold
    pca = PCA(n_components=n_components)
    pca.fit(X_train_fold, y_train_fold)
    X_train_fold_pca = pca.transform(X_train_fold)
    X_val_fold_pca = pca.transform(X_val_fold)

    # Train model on the transformed data
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_fold_pca, y_train_fold)

    # Evaluate model performance on validation data
    fold_score = model.score(X_val_fold_pca, y_val_fold)
    fold_scores.append(fold_score)

  # Average scores across folds for this n_components
  cv_score = np.mean(fold_scores)
  cv_scores_pca.append(cv_score)

# LDA Preprocessing and Cross-Validation

cv_scores_lda = []

for n_components in range(1, min(X_train.shape[1], len(np.unique(y_train)) - 1) + 1):
  # KFold cross-validation
  kf = KFold(n_splits=5, shuffle=True, random_state=42)

  fold_scores = []
  for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # LDA within the fold
    lda = LDA(n_components=n_components)
    lda.fit(X_train_fold, y_train_fold)
    X_train_fold_lda = lda.transform(X_train_fold)
    X_val_fold_lda = lda.transform(X_val_fold)

    # Train model on the transformed data
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_fold_lda, y_train_fold)

    # Evaluate model performance on validation data
    fold_score = model.score(X_val_fold_lda, y_val_fold)
    fold_scores.append(fold_score)

  # Average scores across folds for this n_components
  cv_score = np.mean(fold_scores)
  cv_scores_lda.append(cv_score)

# Choose dimensionality reduction technique based on CV scores
if np.max(cv_scores_pca) > np.max(cv_scores_lda):
  best_technique = "PCA"
  best_cv_score = np.max(cv_scores_pca)
  best_n_components = n_components_list[np.argmax(cv_scores_pca)]
else:
  best_technique = "LDA"
  best_cv_score = np.max(cv_scores_lda)
  best_n_components = n_components_list[np.argmax(cv_scores_lda)]

print("Best dimensionality reduction technique:", best_technique)
print("Best CV score with", best_technique, ":", best_cv_score)

# Training with Best Dimensionality Reduction
if best_technique == "PCA":
  # Use PCA with the chosen number of components
  pca = PCA(n_components=best_n_components)
  pca.fit(X_train, y_train)
  X_train_cv_transformed = pca.transform(X_train)
  X_test_transformed = pca.transform(X_test)
  print("Dimensionality reduction technique chosen:", best_technique)
elif best_technique == "LDA":
  # Use LDA with the chosen number of components
  lda = LDA(n_components=best_n_components)
  lda.fit(X_train, y_train)
  X_train_cv_transformed = lda.transform(X_train)
  X_test_transformed = lda.transform(X_test)
  print("Dimensionality reduction technique chosen:", best_technique)
else:
  # No dimensionality reduction chosen, use original features
  X_train_cv_transformed = X_train.copy()
  X_test_transformed = X_test.copy()
  print("No dimensionality reduction chosen (using original features)")

# Choose the desired number of components for LDA (usually 2 or 3 for visualization)
n_components = 2

# Apply LDA for dimensionality reduction
lda = LDA(n_components=n_components)
X_train_reduced = lda.fit_transform(X_train, y_train)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=50)
X_train_embedded = tsne.fit_transform(X_train_reduced)

# Prepare color mapping
unique_classes = np.unique(y_train)
cmap = plt.cm.get_cmap('tab10')  # Choose a colormap
colors = cmap(np.linspace(0, 1, len(unique_classes)))

# Count data points for each class
class_counts = np.bincount(y_train)

# Create the t-SNE visualization
plt.figure(figsize=(12, 8))
for i, label in enumerate(unique_classes):
    plt.scatter(X_train_embedded[y_train == label, 0], X_train_embedded[y_train == label, 1],
                label=label, c=colors[i], s=class_counts[i] * 0.001)  # Adjust marker size

# Add legend and labels
plt.legend(title='Classes')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('t-SNE Visualization after LDA (CV Score: 0.91)')
plt.axis('off')
plt.show()