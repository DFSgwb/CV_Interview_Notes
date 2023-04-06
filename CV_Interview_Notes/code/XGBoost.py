from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import numpy as np

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dtrain = xgb.DMatrix(X_train, label=y_train)

param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softprob', 'num_class': 3}
num_round = 10

bst = xgb.train(param, dtrain, num_round)

dtest = xgb.DMatrix(X_test)
y_pred = bst.predict(dtest)
classes_x = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, classes_x)
precision = precision_score(y_test, classes_x, average='macro')
recall = recall_score(y_test, classes_x, average='macro')
f1 = f1_score(y_test, classes_x, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
