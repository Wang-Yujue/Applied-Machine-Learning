# Dataset
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

# synthetic dataset for classification (binary)
plt.figure()
plt.title('Sample binary classification problem with two informative features')
X_C2, y_C2 = make_classification(n_samples = 100, n_features=2,
                                n_redundant=0, n_informative=2,
                                n_clusters_per_class=1, flip_y = 0.1,
                                class_sep = 0.5, random_state=0)
plt.scatter(X_C2[:, 0], X_C2[:, 1], c=y_C2,
           marker= 'o', s=50, cmap=cmap_bold)
plt.show()

# Breast cancer dataset for classification
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

# Linear models for classification
# Logistic regression
# Logistic regression for binary classification on fruits dataset using height, width features (positive class: apple, negative class: others)

from sklearn.linear_model import LogisticRegression
from adspy_shared_utilities import (plot_class_regions_for_classifier_subplot)
import KNN_Fruit_2 as Fr

fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
y_fruits_apple = Fr.y_fruits_2d == 1  # make into a binary problem: apples vs everything else
X_train, X_test, y_train, y_test = (
train_test_split(Fr.X_fruits_2d.as_matrix(),
                 y_fruits_apple.as_matrix(),
                 random_state=0))

clf = LogisticRegression(C=100).fit(X_train, y_train)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None,
None, 'Logistic regression \
for binary classification\nFruit dataset: Apple vs others',
subaxes)

h = 6
w = 8
print('A fruit with height {} and width {} is predicted to be: {}'
.format(h, w, ['not an apple', 'an apple'][clf.predict([[h, w]])[0]]))

h = 10
w = 7
print('A fruit with height {} and width {} is predicted to be: {}'
.format(h, w, ['not an apple', 'an apple'][clf.predict([[h, w]])[0]]))
subaxes.set_xlabel('height')
subaxes.set_ylabel('width')

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
.format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
.format(clf.score(X_test, y_test)))

# Logistic regression on simple synthetic dataset

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2,
random_state = 0)

fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
clf = LogisticRegression().fit(X_train, y_train)
title = 'Logistic regression, simple synthetic dataset C = {:.3f}'.format(1.0)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
None, None, title, subaxes)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
.format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
.format(clf.score(X_test, y_test)))

# Logistic regression regularization: C parameter

X_train, X_test, y_train, y_test = (
train_test_split(Fr.X_fruits_2d.as_matrix(),
                 y_fruits_apple.as_matrix(),
                 random_state=0))

fig, subaxes = plt.subplots(3, 1, figsize=(4, 10))

for this_C, subplot in zip([0.1, 1, 100], subaxes):
    clf = LogisticRegression(C=this_C).fit(X_train, y_train)
    title = 'Logistic regression (apple vs rest), C = {:.3f}'.format(this_C)

    plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test, y_test, title, subplot)

plt.tight_layout()

# Application to real dataset

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state=0)

clf = LogisticRegression().fit(X_train, y_train)
print('Breast cancer dataset')
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
.format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
.format(clf.score(X_test, y_test)))

plt.show()