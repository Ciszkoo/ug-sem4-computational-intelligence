import pandas as pd
from sklearn.model_selection import train_test_split
import numpy

df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(
    df.values, train_size=0.7, random_state=278873)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

# display train_set sorted by iris species
# print(train_set[train_set[:, 4].argsort()])

def classify_iris(sl, sw, pl, pw):
    if pw < 1:
        return ("setosa")
    elif pw > 2 or pl > 5:
        return ("virginica")
    else:
        return ("versicolor")


good_predictions = 0
len = test_set.shape[0]

for i in range(len):
    sl, sw, pl, pw = test_inputs[i]
    if classify_iris(sl, sw, pl, pw) == test_set[i, 4]:
        good_predictions = good_predictions + 1

print(good_predictions)
print(good_predictions/len*100, "%")
