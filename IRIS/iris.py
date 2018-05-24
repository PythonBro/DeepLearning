# This program is written in Python3
# https://qiita.com/wtnb93/items/d7a3eb2c3cc0b8c8086b
# This program use the SVM (Support Vector Machine)

from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()

clf = svm.SVC()
clf.fit(iris.data, iris.target)

test_data = [[5.1, 3.5, 1.4, 0.2]]
print(clf.predict(test_data))
