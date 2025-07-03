import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf

'''
Here this program to generate the existing dataset from Available dataset
'''
# data = pd.read_csv(r"C:\Users\Uday Hiremath\Downloads\CNN_DATASET\Chest_XRay_Dataset\Ground_Truth.csv")
# image_dir = r'C:\Users\Uday Hiremath\Downloads\CNN_DATASET\Chest_XRay_Dataset\xray_images'
#
# data['filepath'] = image_dir + '/' + data['Image Index']
#
# # print(Label_maping)
# # print(data["Finding Labels"].unique())
# print(data["Finding Labels"].value_counts()[:15])
#
# data['exists'] = data['filepath'].apply(os.path.exists)
#
#
# missing_count = (~data['exists']).sum()
# print(f"Missing image files: {missing_count}")
#
#
# print(data[~data['exists']].head())
#
# data2 = data[data['exists']]
#
# data2.to_csv(r'C:\Users\Uday Hiremath\Downloads\CNN_DATASET\Chest_XRay_Dataset\Exist_data_set.csv', index=False)

'''
To generate the Labelled dataset
'''

data = pd.read_csv(r'C:\Users\Uday Hiremath\Downloads\CNN_DATASET\Chest_XRay_Dataset\Exist_data_set.csv')

Le = LabelEncoder()

data['Encoded_labels'] = Le.fit_transform(data['Finding Labels'])

Label_maping = dict(zip(Le.classes_, Le.transform(Le.classes_)))

data.to_csv(r'C:\Users\Uday Hiremath\Downloads\CNN_DATASET\Chest_XRay_Dataset\Exist_data_set_labelled.csv', index=False)
