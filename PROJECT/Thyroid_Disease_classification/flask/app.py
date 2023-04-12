# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:56:38 2022

@author: SmartBridge-PC
"""

from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

model = pickle.load(open(r'Training/thyroid_1_model.pkl', 'rb'))
le = pickle.load(open('Training/label_encoder.pkl', 'rb'))


app = Flask(__name__)


@app.route("/")
def about():
    return render_template('home.html')


@app.route("/predict")
def home1():
    return render_template('predict.html')

@app.route("/pred", methods=['POST', 'GET'])
def predict():
    x = []
    for feature in ['goitre','tumor','hypopituitary','psych','TSH','T3','TT4','T4U','FTI','TBG']:
        value = request.form.get(feature, '')
        try:
            x.append(float(value))
        except ValueError:
            x.append(0.0)
    x = [x]
    print(x)
    col = ['goitre','tumor','hypopituitary','psych','TSH','T3','TT4','T4U','FTI','TBG']
    x = pd.DataFrame(x, columns=col)
    # rest of the code

   
    #print(x.shape)

    print(x)
    pred = model.predict(x)
    pred = le.inverse_transform(pred)
    print(pred[0])
    return render_template('submit.html', prediction_text=str(pred))


if __name__ == "__main__":
    app.run(debug=False)
