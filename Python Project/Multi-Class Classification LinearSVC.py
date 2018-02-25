import KNN_Fruit_2 as Fr
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
# Multi - class classification with linear models

# LinearSVC with M classes generates M one vs rest classifiers.

from sklearn.svm import LinearSVC

X_train, X_test, y_train, y_test = train_test_split(Fr.X_fruits_2d, Fr.y_fruits_2d, random_state=0)

clf = LinearSVC(C=5, random_state=67).fit(X_train, y_train)
print('Coefficients:\n', clf.coef_)
print('Intercepts:\n', clf.intercept_)

# Multi - class results on the fruit dataset

plt.figure(figsize=(6, 6))
colors = ['r', 'g', 'b', 'y']
cmap_fruits = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00'])

plt.scatter(Fr.X_fruits_2d[['height']], Fr.X_fruits_2d[['width']],
            c=Fr.y_fruits_2d, cmap=cmap_fruits, edgecolor='black', alpha=.7)

x_0_range = np.linspace(-10, 15)

for w, b, color in zip(clf.coef_, clf.intercept_, ['r', 'g', 'b', 'y']):
    # Since class prediction with a linear model uses the formula y = w_0 x_0 + w_1 x_1 + b,
    # and the decision boundary is defined as being all points with y = 0, to plot x_1 as a
    # function of x_0 we just solve w_0 x_0 + w_1 x_1 + b = 0 for x_1:
    plt.plot(x_0_range, -(x_0_range * w[0] + b) / w[1], c=color, alpha=.8)

plt.legend(Fr.target_names_fruits)
plt.xlabel('height')
plt.ylabel('width')
plt.xlim(-2, 12)
plt.ylim(-2, 15)
plt.show()