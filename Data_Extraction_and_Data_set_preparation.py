import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf
data = pd.read_csv(r"C:\Users\Uday Hiremath\Downloads\CNN_DATASET\Chest_XRay_Dataset\Ground_Truth.csv")
image_dir = r'C:\Users\Uday Hiremath\Downloads\CNN_DATASET\Chest_XRay_Dataset\xray_images'

data['filepath'] = image_dir + '/' + data['Image Index']

# print(Label_maping)
# print(data["Finding Labels"].unique())
print(data["Finding Labels"].value_counts()[:15])

data['exists'] = data['filepath'].apply(os.path.exists)


missing_count = (~data['exists']).sum()
print(f"Missing image files: {missing_count}")


print(data[~data['exists']].head())

data2 = data[data['exists']]

data2.to_csv(r'C:\Users\Uday Hiremath\Downloads\CNN_DATASET\Chest_XRay_Dataset\Exist_data_set.csv', )

data3 = pd.read_csv(r'C:\Users\CMTI\Downloads\dataset for ML modeling\Chest_XRay_Dataset\Exist_data_set.csv')

Le = LabelEncoder()

data3['Encoded_labels'] = Le.fit_transform(data3['Finding Labels'])

Label_maping = dict(zip(Le.classes_, Le.transform(Le.classes_)))

data3.to_csv(r'C:\Users\CMTI\Downloads\dataset for ML modeling\Chest_XRay_Dataset\Exist_data_set_labelled.csv')
data = pd.read_csv(r'C:\Users\CMTI\Downloads\dataset for ML modeling\Chest_XRay_Dataset\Exist_data_set_labelled.csv')
print(data['Encoded_labels'].value_counts()[:25])
print(data['Finding Labels'].value_counts()[:25])

filepaths = data['filepaths']