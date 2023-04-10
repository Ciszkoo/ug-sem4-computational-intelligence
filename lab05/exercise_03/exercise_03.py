import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv("iris.csv")

all_inputs = df[["sepal.length", "sepal.width", "petal.length", "petal.width"]].values
all_classes = df["species"]

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=278873)

dtc = DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)
accuracy = dtc.score(test_inputs, test_classes)

print(accuracy, "%")

titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true")
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        dtc,
        test_inputs,
        test_classes,
        normalize=normalize
    )
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)