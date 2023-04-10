from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

iris = load_iris()

train_data, test_data, train_labels, test_labels = train_test_split(iris.data, iris.target, test_size=0.7)

scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

print(train_data[:3])

# ---------------------------------------------------------------- #
print("Dwa neurony w warstwie ukrytej")

mlp = MLPClassifier(hidden_layer_sizes=(2), max_iter=2000)
mlp.fit(train_data, train_labels)

predictions_train = mlp.predict(train_data)
print(accuracy_score(predictions_train, train_labels))
predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_labels))

train_confusion_matrix = confusion_matrix(predictions_train, train_labels)
print(train_confusion_matrix)
test_confusion_matrix = confusion_matrix(predictions_test, test_labels)
print(test_confusion_matrix)

print(classification_report(predictions_test, test_labels))

# ---------------------------------------------------------------- #
print("Trzy neurony w warstwie ukrytej")

mlp = MLPClassifier(hidden_layer_sizes=(3), max_iter=2000)
mlp.fit(train_data, train_labels)

predictions_train = mlp.predict(train_data)
print(accuracy_score(predictions_train, train_labels))
predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_labels))

train_confusion_matrix = confusion_matrix(predictions_train, train_labels)
print(train_confusion_matrix)
test_confusion_matrix = confusion_matrix(predictions_test, test_labels)
print(test_confusion_matrix)

print(classification_report(predictions_test, test_labels))

# ---------------------------------------------------------------- #
print("Dwie warstwy ukryte po trzy neurony")

mlp = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=2000)
mlp.fit(train_data, train_labels)

predictions_train = mlp.predict(train_data)
print(accuracy_score(predictions_train, train_labels))
predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_labels))

train_confusion_matrix = confusion_matrix(predictions_train, train_labels)
print(train_confusion_matrix)
test_confusion_matrix = confusion_matrix(predictions_test, test_labels)
print(test_confusion_matrix)

print(classification_report(predictions_test, test_labels))