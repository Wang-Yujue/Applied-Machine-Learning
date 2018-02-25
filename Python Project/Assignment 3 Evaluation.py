import numpy as np
import pandas as pd

# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def answer_one():

    per = (df['Class'] == 1).value_counts(True)

    return per[1]


# print(answer_one())


def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score

    # Negative class (0) is most frequent
    dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    # Therefore the dummy 'most_frequent' classifier always predicts class 0
    y_dummy_predictions = dummy_majority.predict(X_test)
    accuracy = dummy_majority.score(X_test, y_test)
    recall = recall_score(y_test, y_dummy_predictions)

    return accuracy, recall


# print(answer_two())


def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC

    svm = SVC().fit(X_train, y_train)
    y_predictions = svm.predict(X_test)
    accuracy = svm.score(X_test, y_test)
    precision = precision_score(y_test, y_predictions)
    recall = recall_score(y_test, y_predictions)

    return accuracy, recall, precision


# print(answer_three())


def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    svm = SVC(C=1e9, gamma=1e-07).fit(X_train, y_train)
    svm_predicted = svm.predict(X_test)
    y_scores = svm.decision_function(X_test)

    # threshold of - 220 on the decision function
    index = np.argwhere(y_scores >= -220)
    svm_predicted[index] = 1
    # svm_predicted = [1 if score > -220 else 0 for score in y_scores]

    confusion = confusion_matrix(y_test, svm_predicted)

    return confusion


# print(answer_four())


def answer_five():
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression().fit(X_train, y_train)
    y_scores_lr = lr.fit(X_train, y_train).decision_function(X_test)

    # Precision-recall curves
    from sklearn.metrics import precision_recall_curve
    from matplotlib import pyplot as plt

    precision, recall, _ = precision_recall_curve(y_test, y_scores_lr)

    index_recall = np.argmin(np.abs(precision - 0.75))
    value_recall = recall[index_recall]

    plt.figure()
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.plot(0.75, value_recall, 'o', c='r')
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)

    # ROC curves, Area-Under-Curve (AUC)
    from sklearn.metrics import roc_curve, auc

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores_lr)

    index_tpr = np.argmin(np.abs(fpr_lr - 0.16))
    value_tpr = tpr_lr[index_tpr]

    roc_auc_lr = auc(fpr_lr, tpr_lr)

    plt.figure()
    plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    # plt.axes().set_aspect('equal')
    # plt.show()

    return value_recall, value_tpr


# print(answer_five())


def answer_six():
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    # Grid search
    lr = LogisticRegression().fit(X_train, y_train)
    grid_values = {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(lr, param_grid=grid_values, scoring='recall')
    grid.fit(X_train, y_train)
    grid_mean_test_score = grid.cv_results_['mean_test_score'].reshape(5,2)

    return grid_mean_test_score


# print(answer_six())


# Use the following function to help visualize results from the grid search
def gridsearch_heatmap(scores):

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure()
    sns.heatmap(scores, xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0)
    plt.show()


# gridsearch_heatmap(answer_six())