from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import json
import torch

sys.path.insert(0,'D:/20211/KHDL/KHDL_IT4930/inference/.')

from misc.ann_model import Model

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Load các thông tin cần thiết
features = json.load(open('D:/20211/KHDL/KHDL_IT4930/inference/misc/all_features.json', "r", encoding='utf-8'))
categorical_features = json.load(open('D:/20211/KHDL/KHDL_IT4930/inference/misc/categorical_features.json', "r", encoding='utf-8'))
stats = pd.read_csv('D:/20211/KHDL/KHDL_IT4930/inference/misc/stats.csv', sep = '\t', index_col=0)
stats_none_price = stats.drop(index=['gia'], axis=1)
data = pd.read_csv('D:/20211/KHDL/KHDL_IT4930/dataset/data.csv', sep = '\t', index_col=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device = device)
load_model = torch.load('D:/20211/KHDL/KHDL_IT4930/inference/misc/ann_162.pt', map_location=device)
model.load_state_dict(load_model['model'])

print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html', cat = categorical_features)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        input = np.zeros((len(features)-1,))
        for key, value in json.loads(request.data.decode('utf8')).items():
            if type(value) == int:
                input[features[key]] = value
            else:
                input[features[key+'_'+value] - 1] = 1
        input = ( input - np.array(list(stats_none_price['mean'])) )/np.array(list(stats_none_price['std']))
        input = torch.tensor(input, dtype = torch.float).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            pred = model(input).squeeze().tolist()
            pred = (pred * stats.loc['gia','std']) + stats.loc['gia','mean']
            return str(pred)
    return None

@app.route('/all_features')
def all_features():
    return features

@app.route('/categorical_feature', methods=['GET','POST'])
def categorical_feature():
    return categorical_features

if __name__ == '__main__':
    app.run(debug=True)

