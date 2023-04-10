import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("diabetes.csv")
print(df.describe().transpose())

target_column = 'class'

df[target_column] = df[target_column].map(lambda x: 1 if  x == 'tested_positive' else 0)

predictors = list(set(list(df.columns)) - set([target_column]))
df[predictors] = df[predictors] / df[predictors].max()

all_inputs = df[predictors].values
all_labels = df[target_column].values

train_inputs, test_inputs, train_labels, test_labels = train_test_split(all_inputs, all_labels, test_size=0.3, random_state=40)

mlp = MLPClassifier(hidden_layer_sizes=(6, 3), activation='relu', solver='adam', max_iter=500)
mlp.fit(train_inputs, train_labels)

predict_train = mlp.predict(train_inputs)
predict_test = mlp.predict(test_inputs)

print(confusion_matrix(train_labels, predict_train))
print(classification_report(train_labels, predict_train))

print(confusion_matrix(test_labels, predict_test))
print(classification_report(test_labels, predict_test))