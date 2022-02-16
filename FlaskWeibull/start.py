#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 19:41:01 2021

@author: wji
"""
debug = True
import os
import sys
import re
import numpy as np
import pandas as pd
from flask import Flask
from flask import render_template
from flask import Response
from flask import jsonify
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from pkg_resources import resource_filename

import utils.weibull as weibull

root_dir_path = resource_filename(__name__, '/')
image_dir_path = resource_filename(__name__, '/image')

print('name:', __name__)
print('root dir path is:', root_dir_path)
print('image dir path is:', image_dir_path)

print(os.path.join(root_dir_path, 'image') )


app = Flask(__name__, template_folder='public',static_folder='static')
app.config["DEBUG"] = True # allow auto reload after code changed


@app.route('/', methods=['GET'])
def home():
    filename = "home.html"
    content = {
    "head": "passing a head",
    "val": "passing a val"
    }
    return render_template(filename, content=content)

@app.route('/test', methods=['GET'])
def test():
    filename = "test.html"
    content = {
    }
    return render_template(filename, content=content)

@app.route('/api/<name>', methods=['GET'])
def api_render(name):
    filename = "api.html"
    content = {
        "title":name
    }
    return render_template(filename, content=content)

# prepare data 
filename = resource_filename(__name__, '/data/data_sim.csv')
df = pd.read_csv(filename)
threshold = 3000
failures = list(df.loc[df['right_censored']==1, 'failures'])
right_censored = list(df.loc[df['right_censored']==0, 'failures'])

@app.route('/weibull', methods=['GET','POST'])
def app_weibull():
    content = weibull.process_weibull(failures=failures, right_censored=right_censored, CI = .95, t = np.linspace(100,5000,101), prefix='test_')
    filename = "weibull.html"
    content = {
    "head": "Weibull plots",
    "val": ""
    }
    return render_template(filename, content=content)

if __name__=="__main__":
    app.run()



#    filename = os.path.join(app.instance_path, 'public', 'home.html')
