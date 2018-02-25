import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler


train = pd.read_csv("train.csv", encoding='cp1252', low_memory=False)
y_train = train['compliance'].fillna(value=0)  # treat null as not paid
# X_train = train[['ticket_id', 'fine_amount', 'admin_fee',
#                  'state_fee', 'late_fee', 'discount_amount', 'clean_up_cost',
#                  'judgment_amount']].fillna(value=0)
# X_train = train[['ticket_id', 'fine_amount', 'admin_fee',
#                  'state_fee', 'late_fee', 'discount_amount', 'clean_up_cost',
#                  'judgment_amount']].dropna(axis=1, how='any')
X_train = train[['ticket_id', 'admin_fee',  # there are NaN in fine_amount only
                 'state_fee', 'late_fee', 'discount_amount', 'clean_up_cost',
                 'judgment_amount']]

test = pd.read_csv('test.csv')
# X_test = test[['ticket_id', 'fine_amount', 'admin_fee',
#                'state_fee', 'late_fee', 'discount_amount', 'clean_up_cost',
#                'judgment_amount']].fillna(value=0)
X_test = test[['ticket_id', 'admin_fee',
               'state_fee', 'late_fee', 'discount_amount', 'clean_up_cost',
               'judgment_amount']]

# Important!!
# important to introduce the normalization to deal with verious feature scales
X_train_scaled = MinMaxScaler().fit_transform(X_train)
X_test_scaled = MinMaxScaler().fit_transform(X_test)

# logistic regression
# For better fit, bigger values specify weaker regularization
lr = LogisticRegression(C=10).fit(X_train_scaled, y_train)
train_score = lr.score(X_train_scaled, y_train)
print(train_score)

y_proba_lr = lr.predict_proba(X_test_scaled)

# output the probability that each corresponding ticket will be paid (positive class)
compliance = pd.Series(y_proba_lr[:, 1], X_test['ticket_id'])


# # SVM is too expensive
# svm = SVC().fit(X_train, y_train)
# train_score = svm.score(X_train, y_train)
# print(train_score)


# # Naive Bayes Classifier, Highly efficient but nad generalization
# nb = GaussianNB().fit(X_train, y_train)
# train_score = nb.score(X_train, y_train)
# print(train_score)  # tend to overfit, because features are many
#
# y_proba_nb = nb.predict_proba(X_test)
#
# # output the probability that each corresponding ticket will be paid (positive class)
# compliance = pd.Series(y_proba_nb[:, 1], X_test['ticket_id'])
