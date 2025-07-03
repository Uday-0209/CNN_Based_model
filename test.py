import pandas as pd
import numpy as np

data = pd.read_csv(r'C:\Users\Uday Hiremath\Downloads\CNN_DATASET\Chest_XRay_Dataset\Ground_Truth.csv')

print(len(data['Image Index'].value_counts()))

