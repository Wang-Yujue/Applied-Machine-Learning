# Preamble and Datasets

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Breast cancer dataset
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y=True)

# Our sample fruits dataset
fruits = pd.read_table('fruit_data_with_colors.txt')
X_fruits = fruits[['mass','width','height', 'color_score']]
y_fruits = fruits[['fruit_label']] - 1


# Multidimensional scaling (MDS) on the fruit dataset
from adspy_shared_utilities import plot_labelled_scatter
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS

# each feature should be centered (zero mean) and with unit variance
X_fruits_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)

mds = MDS(n_components=2)

X_fruits_mds = mds.fit_transform(X_fruits_normalized)

# plot_labelled_scatter(X_fruits_mds, y_fruits, ['apple', 'mandarin', 'orange', 'lemon'])
# plt.xlabel('First MDS feature')
# plt.ylabel('Second MDS feature')
# plt.title('Fruit sample dataset MDS')

# Multidimensional scaling (MDS) on the breast cancer dataset (compare it to the results from PCA)
# each feature should be centered (zero mean) and with unit variance
X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)

mds = MDS(n_components=2, random_state=0)

X_mds = mds.fit_transform(X_normalized)

plot_labelled_scatter(X_mds, y_cancer, ['malignant', 'benign'])

plt.xlabel('First MDS dimension')
plt.ylabel('Second MDS dimension')
plt.title('Breast Cancer Dataset MDS (n_components = 2)')

# t-SNE on the fruit dataset (you can see how some dimensionality reduction methods may be less successful on some datasets.
# Here, it doesn't work as well at finding structure in the small fruits dataset, compared to other methods like MDS)
from sklearn.manifold import TSNE

# tsne = TSNE(random_state = 0)
#
# X_tsne = tsne.fit_transform(X_fruits_normalized)

# plot_labelled_scatter(X_tsne, y_fruits,
#     ['apple', 'mandarin', 'orange', 'lemon'])
# plt.xlabel('First t-SNE feature')
# plt.ylabel('Second t-SNE feature')
# plt.title('Fruits dataset t-SNE')

# t-SNE on the breast cancer dataset (See the reading "How to Use t-SNE effectively" for further details on how
# the visualizations from t-SNE are affected by specific parameter settings)
tsne = TSNE(random_state = 0)

X_tsne = tsne.fit_transform(X_normalized)

plot_labelled_scatter(X_tsne, y_cancer,
    ['malignant', 'benign'])
plt.xlabel('First t-SNE feature')
plt.ylabel('Second t-SNE feature')
plt.title('Breast cancer dataset t-SNE')