# Weibull Analysis

# How to run
#### compile module
`python setup.py install`
#### start app
`python FlaskWeibull/start.py`

#### Usage: 
1) input with form: http://0.0.0.0:3000/
2) input with json: http://0.0.0.0:3000/weibull + json_input 
##### example json: 
`{ "failures" : [12,23,43,34], "right_censored" : [] }`