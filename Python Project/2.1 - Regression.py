import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(0)
n = 15
x = np.linspace(0, 10, n) + np.random.randn(n) / 5
y = np.sin(x) + x / 6 + np.random.randn(n) / 10

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# visualize the dataset
def part1_scatter():
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);
    plt.show()

# part1_scatter()

def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    X = x.reshape(-1, 1)
    X_hat = np.linspace(0, 10, 100).reshape(-1, 1)
    y_hat = np.zeros([4, 100])
    a = 0
    for i in [1, 3, 6, 9]:
        poly = PolynomialFeatures(degree=i)
        X_poly = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=0)
        linreg = LinearRegression().fit(X_train, y_train)
        X_hat_poly = poly.fit_transform(X_hat)
        y_hat[a] = linreg.predict(X_hat_poly)
        a = a + 1
    return y_hat

def plot_one(degree_predictions):
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)

# plot_one(answer_one())
# plt.show()

def answer_two():
    from sklearn.metrics.regression import r2_score
    X = x.reshape(-1, 1)
    r2_train = np.zeros([10, ])
    r2_test = np.zeros([10, ])
    a = 0
    for i in range(0, 10):
        poly = PolynomialFeatures(degree=i)
        X_poly = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=0)
        linreg = LinearRegression().fit(X_train, y_train)

        # Make predictions
        y_pred_test = linreg.predict(X_test)
        y_pred_train = linreg.predict(X_train)
        # r-squared score
        r2_test[a] = r2_score(y_test, y_pred_test)
        r2_train[a] = r2_score(y_train, y_pred_train)

        a = a + 1
    scr = (r2_train, r2_test)
    return scr

# print(answer_two())

def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression

    X = x.reshape(-1, 1)
    poly = PolynomialFeatures(degree=12)
    X_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=0)
    linreg = LinearRegression().fit(X_train, y_train)

    print('(poly deg 12) R-squared score (test): {:.3f}\n'
          .format(linreg.score(X_test, y_test)))
    LinearRegression_R2_test_score = linreg.score(X_test, y_test)

    # Lasso regulation
    linreg = Lasso(alpha=0.01, max_iter=10000).fit(X_train, y_train)
    print('(poly deg 12 + Lasso) R-squared score (test): {:.3f}'
          .format(linreg.score(X_test, y_test)))
    Lasso_R2_test_score = linreg.score(X_test, y_test)

    return (LinearRegression_R2_test_score, Lasso_R2_test_score)
# print(answer_four())