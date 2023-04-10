import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("iris.csv")

all_inputs = df[["sepal.length", "sepal.width", "petal.length", "petal.width"]].values
all_classes = df["species"]

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=278873)

# DTC
dtc = DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)
dtcAccuracy = dtc.score(test_inputs, test_classes)

# GNB
gnb = GaussianNB()
gnb.fit(train_inputs, train_classes)
gnbAccuracy = gnb.score(test_inputs, test_classes)

# KNN 3
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(train_inputs, train_classes)
knn3Accuracy = knn3.score(test_inputs, test_classes)

# KNN 5
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(train_inputs, train_classes)
knn5Accuracy = knn5.score(test_inputs, test_classes)

# KNN 11
knn11 = KNeighborsClassifier(n_neighbors=11)
knn11.fit(train_inputs, train_classes)
knn11Accuracy = knn11.score(test_inputs, test_classes)

print(f"Decision Tree accuracy: {dtcAccuracy}%")
print(f"Naive Bayes accuracy: {gnbAccuracy}%")
print(f"K-Nearest Neighbors 3 accuracy: {knn3Accuracy}%")
print(f"K-Nearest Neighbors 5 accuracy: {knn5Accuracy}%")
print(f"K-Nearest Neighbors 11 accuracy: {knn11Accuracy}%")