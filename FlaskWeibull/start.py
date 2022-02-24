#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Wei Ji
# Usage: curl -k -d '{ "failures" : [12,23,43,34], "right_censored" : [] } ' https://<IPAddress>/weibull
"""
Created on Thu Feb 25 19:41:01 2021

@author: wji
"""
debug = False

import os
import sys
import re
import numpy as np
import pandas as pd
import json
from flask import Flask, Response, request
from flask import render_template, jsonify,  make_response, redirect, url_for 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from pkg_resources import resource_filename

import utils.weibull as weibull

root_dir_path = resource_filename(__name__, '/')
image_dir_path = resource_filename(__name__, '/image')

if debug:
    print('name:', __name__)
    print('root dir path is:', root_dir_path)
    print('image dir path is:', image_dir_path)

    print(os.path.join(root_dir_path, 'image') )

app = Flask(__name__, template_folder='public', static_folder='static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["DEBUG"] = True # allow auto reload after code changed


ex_failures = [10,34,16,36,50]
ex_right_censored = [20,30]

@app.route('/', methods=['GET', 'POST'])
def home():
    filename = "home.html"
    content = {
    "head": "Home",
    "val": ""
    }
    return render_template(filename, content=content)

def split_list_num(s: str) -> list:
    arr = re.split('[\s,]+', s)
    if len(arr)==0 or arr[0]=='':
        return []
    else:
        res = [float(a) for a in arr]
        return res
    return []

@app.route('/weibull', methods=['GET', 'POST'])
def app_weibull():
    failures = []
    right_censored = []
    head = ''
    json_data = None

    # Validate the request body contains JSON
    if request.is_json:
        # Parse the JSON into a Python dictionary
        json_data = request.get_json()
    elif request.form.get('failures') is not None:
        json_data = {
            'failures': split_list_num(request.form.get('failures')),
            'right_censored': split_list_num(request.form.get('right_censored')),
        }    
    else:
        json_data = {
            'failures': failures,
            'right_censored': right_censored,
        }

    if json_data is None or 'failures' not in json_data \
            or len(json_data['failures'])==0 \
            or np.min(json_data['failures']) <=0 :

        failures = ex_failures
        right_censored = ex_right_censored
        head = 'Showing Weibull plots examples'
        filename = "weibull.html"
        content = {
        "head": head,
        "val": ""
        }
        return render_template(filename, content=content)
    else:
        failures = json_data['failures']
        right_censored = json_data.get('right_censored', [])
        head = "Weibull plots"
        t_min = np.max([np.min(failures), 0,]) + 1.e-5
        t_max = np.max(failures) * 2.
        if len(right_censored)>0:
            t_max = np.max([t_max, np.max(right_censored) * 2.])
        t = np.linspace(t_min, t_max, 101, endpoint=True)
        weibull.process_weibull(failures=failures, right_censored=right_censored, CI = .95, t = t, prefix='test_')
        filename = "weibull.html"
        content = {
        "head": head,
        "val": ""
        }
        return render_template(filename, content=content)

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-evalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__=="__main__":
    app.run(host='0.0.0.0', port=3000)