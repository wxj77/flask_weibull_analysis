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
from flask import Flask
from flask import render_template
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from pkg_resources import resource_filename
root_dir_path = resource_filename(__name__, '/')
image_dir_path = resource_filename(__name__, '/image')

print('name:', __name__)
print('name:', __file__)
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

if __name__=="__main__":
    app.run()



#    filename = os.path.join(app.instance_path, 'public', 'home.html')
