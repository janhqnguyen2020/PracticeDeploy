from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os

# Import needed packages for classification
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = '13e75254682c41b0b263d25101d177fa'

@app.route('/')
def index():
    return render_template('submission.html')

@app.route('/prediction', methods = ['POST', 'GET'])
def prediction():
    pregnancy_num = request.form['pregnancy_num']
    glucose_level = request.form['glucose_level']
    blood_pressure = request.form['blood_pressure']
    skin_thickness = request.form['skin_thickness']
    insulin_level = request.form['insulin_level']
    BMI = request.form['BMI']
    DBF = request.form['DBF']
    age = request.form['age']

    unpickled_scaler = pickle.load(open('scalerObject.pkl', 'rb'))

    unpickled_KNeighborModel = pickle.load(open('classifierModel.pkl', 'rb'))

    user_list = [[pregnancy_num, glucose_level, blood_pressure, skin_thickness, insulin_level, BMI, DBF, age]]
    user_list = unpickled_scaler.transform(user_list)

    cancer_prediction = unpickled_KNeighborModel.predict(user_list)

    if (cancer_prediction[0] == 1):
        pred = "Cancer Detected"
    else:
        pred = "Cancer Not Detected"

    return render_template('prediction.html', prediction = pred)

if __name__ == "__main__":
    app.run(debug=True)
