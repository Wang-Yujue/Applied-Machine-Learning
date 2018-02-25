# Dataset
from sklearn.datasets import load_breast_cancer
# Breast cancer dataset for classification
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

# Decision Trees
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
from sklearn.model_selection import train_test_split


iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=3)
clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))

# Setting max decision tree depth to help avoid overfitting

clf2 = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf2.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
      .format(clf2.score(X_test, y_test)))

# Visualizing decision trees
# plot_decision_tree(clf, iris.feature_names, iris.target_names)

# Pre - pruned version(max_depth=3)
plot_decision_tree(clf2, iris.feature_names, iris.target_names)

# Feature importance
from adspy_shared_utilities import plot_feature_importances

plt.figure(figsize=(10, 4), dpi=80)
plot_feature_importances(clf, iris.feature_names)
plt.show()

print('Feature importances: {}'.format(clf.feature_importances_))


from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

pair_list = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
tree_max_depth = 4

for pair, axis in zip(pair_list, subaxes):
    X = X_train[:, pair]
    y = y_train

    clf = DecisionTreeClassifier(max_depth=tree_max_depth).fit(X, y)
    title = 'Decision Tree, max_depth = {:d}'.format(tree_max_depth)
    plot_class_regions_for_classifier_subplot(clf, X, y, None,
                                              None, title, axis,
                                              iris.target_names)

    axis.set_xlabel(iris.feature_names[pair[0]])
    axis.set_ylabel(iris.feature_names[pair[1]])

plt.tight_layout()
plt.show()

# Decision Trees on a real - world dataset

from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
from adspy_shared_utilities import plot_feature_importances

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state=0)

clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=8,
                             random_state=0).fit(X_train, y_train)

plot_decision_tree(clf, cancer.feature_names, cancer.target_names)

print('Breast cancer dataset: decision tree')
print('Accuracy of DT classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of DT classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))

plt.figure(figsize=(10, 6), dpi=80)
plot_feature_importances(clf, cancer.feature_names)
plt.tight_layout()

plt.show()
print('Breast cancer dataset: decision tree')
print('Accuracy of DT classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of DT classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))

plt.figure(figsize=(10, 6), dpi=80)
plot_feature_importances(clf, cancer.feature_names)
plt.tight_layout()

plt.show()