# Import needed packages for classification
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

import warnings
warnings.filterwarnings("ignore")

unpickled_scaler = pickle.load(open('scalerObject.pkl', 'rb'))

unpickled_KNeighborModel = pickle.load(open('classifierModel.pkl', 'rb'))

ex_list = [[6, 148.0, 72.0, 35.0, 155.0, 33.6, 0.627, 50]]
ex_list = unpickled_scaler.transform(ex_list)

prediction = unpickled_KNeighborModel.predict(ex_list)

if (prediction[0] == 1):
    print("Cancer Detected")
else:
    print("Cancer Not Detected")
