# pip install -e .
# python version 3.7
from setuptools import setup, find_packages

setup(
   name='FlaskWeibull',
   version='1.0',
   description='A flask module',
   author='Wei Ji',
   author_email='jiwei0706@gmail.com',
   packages=find_packages(include=['*']),  #same as name
   
   install_requires=['wheel==0.36.2',
                     'flask==1.1.1',
                     'matplotlib==3.5.1',
                     'autograd==1.3',
                     'autograd-gamma==0.5.0',
                     'docutils==0.15',
                     'mplcursors==0.5.1',
                     'scikit-learn==0.22.1',
                     'scipy==1.7.3',
                     'reliability==0.8.1',
                     'pandas==1.1.2',
                     'numpy==1.19.5'], #external packages as dependencies
)

print('packages', find_packages(include=['.','*'])),