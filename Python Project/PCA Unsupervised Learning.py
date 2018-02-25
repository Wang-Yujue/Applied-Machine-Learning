# Preamble and Datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Breast cancer dataset
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

# Our sample fruits dataset
fruits = pd.read_table('fruit_data_with_colors.txt')
X_fruits = fruits[['mass','width','height', 'color_score']]
y_fruits = fruits[['fruit_label']] - 1


# Principal Components Analysis (PCA)

# Using PCA to find the first two principal components of the breast cancer dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Before applying PCA, each feature should be centered (zero mean) and with unit variance
X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)

pca = PCA(n_components = 2).fit(X_normalized)

X_pca = pca.transform(X_normalized)
print(X_cancer.shape, X_pca.shape)

# Plotting the PCA-transformed version of the breast cancer dataset
from adspy_shared_utilities import plot_labelled_scatter
plot_labelled_scatter(X_pca, y_cancer, ['malignant', 'benign'])

plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Breast Cancer Dataset PCA (n_components = 2)')

# Plotting the magnitude of each feature value for the first two principal components
fig = plt.figure(figsize=(8, 4))
plt.imshow(pca.components_, interpolation='none', cmap='plasma')
feature_names = list(cancer.feature_names)

plt.gca().set_xticks(np.arange(-.5, len(feature_names)))
plt.gca().set_yticks(np.arange(0.5, 2))
plt.gca().set_xticklabels(feature_names, rotation=90, ha='left', fontsize=12)
plt.gca().set_yticklabels(['First PC', 'Second PC'], va='bottom', fontsize=12)

plt.colorbar(orientation='horizontal', ticks=[pca.components_.min(), 0,
                                              pca.components_.max()], pad=0.65)

# PCA on the fruit dataset (for comparison)

# each feature should be centered (zero mean) and with unit variance
X_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)

pca = PCA(n_components=2).fit(X_normalized)
X_pca = pca.transform(X_normalized)

# plot_labelled_scatter(X_pca, y_fruits, ['apple','mandarin','orange','lemon'])
#
# plt.xlabel('First principal component')
# plt.ylabel('Second principal component')
# plt.title('Fruits Dataset PCA (n_components = 2)')

