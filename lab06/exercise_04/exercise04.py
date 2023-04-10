import pandas as pd
import tensorflow as tf

df = pd.read_csv("diabetes.csv")

target_name = 'class'
df[target_name] = df[target_name].map(
    lambda x: 1 if x == 'tested_positive' else 0)

target = df.pop(target_name)

numeric_feature_name = ['pregnant-times', 'glucose-concentr', 'blood-pressure',
                        'skin-thickness', 'insulin', 'mass-index', 'pedigree-func', 'age']
numeric_features = df[numeric_feature_name]

tf.convert_to_tensor(numeric_features)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(numeric_features)

# normalizer(numeric_features.iloc[:3])

def get_basic_model():
  model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = get_basic_model()
model.fit(numeric_features, target, epochs=15, batch_size=2)