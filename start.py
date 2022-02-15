#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 19:41:01 2021

@author: wji
"""
import os, sys, re
from flask import Flask
from flask import render_template

app = Flask(__name__, template_folder='public',static_folder='static')
app.config["DEBUG"] = True # allow auto reload after code changed

#@app.route('/')
#def hello_world():
#    return 'Hello, World. Wei Ji!'
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
