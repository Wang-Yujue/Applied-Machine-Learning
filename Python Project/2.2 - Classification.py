import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2

def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    # important note: random_state=0 to ensure reproducibility
    clf = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    # Feature importance

    plt.figure()
    c_features = len(X_train2.columns)
    plt.barh(range(c_features), clf.feature_importances_)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature name")
    plt.yticks(np.arange(c_features), X_train2.columns)

    #print('Feature importances: {}'.format(clf.feature_importances_))

    feature_importance_index = np.argsort(clf.feature_importances_)[::-1]
    Feature_name = list(X_train2.columns[feature_importance_index[0:5]])

    return Feature_name

# print(answer_five())
# plt.show()

def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    # Support Vector Machine with RBF kernel: gamma parameter
    # Validation curve example

    param_range = np.logspace(-4, 1, 6)
    train_scores, test_scores = validation_curve(SVC(kernel='rbf', C=1, random_state=0), X_subset, y_subset,
                                                 param_name='gamma',
                                                 param_range=param_range, cv=3, scoring='accuracy')

    # print(train_scores)
    # print(test_scores)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    return (train_scores_mean, test_scores_mean)
#print(answer_six())