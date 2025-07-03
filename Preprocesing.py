import numpy as np
import pandas as pd
import  cv2 as cv
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
warnings.filterwarnings('ignore')

data = pd.read_csv(r'C:\Users\Uday Hiremath\Downloads\CNN_DATASET\Chest_XRay_Dataset\Exist_data_set.csv')

# print(data)
#
# print(data.columns)
'''
Index(['Image Index', 'Finding Labels', 'Patient ID', 'Patient Age',
       'Patient Gender', 'View Position', 'filepath', 'exists'] 
       '''

print(data['Finding Labels'].value_counts()[:20])
data["Finding Labels"] = data["Finding Labels"].str.split('|')
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(data["Finding Labels"])
print(labels)
#joblib.dump(mlb,r"C:\Users\Uday Hiremath\Downloads\CNN_DATASET\Chest_XRay_Dataset\mlb_labelled.pkl")

classes = mlb.classes_
print(classes)
label_maping = dict(zip(mlb.classes_, mlb.transform(mlb.classes_)))
print(label_maping)

filepaths = data['filepath'].values

dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

def preprocess_image(filepaths, labels):
    image = tf.io.read_file(filepaths)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image/255.0
    noise = tf.random.normal(shape = tf.shape(image), mean = 0.0, stddev = 0.5, dtype = tf.float32)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return image, labels

dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
total_size = len(dataset)
train_size = int(0.8 * len(data))
train_ds = dataset.take(train_size).batch(90).prefetch(tf.data.AUTOTUNE)
test_ds = dataset.skip(train_size).batch(90).prefetch(tf.data.AUTOTUNE)
'''Able to achieve accuracy of 0.54'''
#
# model = models.Sequential([
#     layers.Input(shape = (224,224,3)),
#     layers.Conv2D(32, (3*3), activation = 'relu'),
#     layers.MaxPooling2D(),
#
#     layers.Conv2D(64, (3*3), activation = 'relu'),
#     layers.MaxPooling2D(),
#
#     layers.Conv2D(128, (3*3), activation = 'relu'),
#     layers.MaxPooling2D(),
#
#     layers.Flatten(),
#     layers.Dense(128, activation = 'relu'),
#     layers.Dropout(0.3),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(len(mlb.classes_), activation = 'sigmoid')
# ])
base_model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
base_model.trainable = True
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation = 'tanh'),
    layers.Dropout(0.5),
    layers.Dense(128, activation = 'tanh'),
    layers.Dropout(0.2),
    layers.Dense(64, activation = 'tanh'),
    layers.Dense(len(mlb.classes_), activation = 'sigmoid')
])
model.compile(optimizer =tf.keras.optimizers.Adam(0.001), loss = 'binary_crossentropy', metrics =['accuracy'])
model.fit(train_ds, validation_data = test_ds, epochs = 5)

loss, acc = model.evaluate(test_ds)
print(f"Validation Accuracy: {acc:.2f}")


