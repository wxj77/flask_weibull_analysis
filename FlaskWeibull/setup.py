from setuptools import setup, find_packages

setup(
   name='FlaskWeibull',
   version='1.0',
   description='A flask module',
   author='Wei Ji',
   author_email='jiwei0706@gmail.com',
   packages=find_packages(include=['*']),  #same as name
   
   install_requires=['wheel',
                    'PyYAML',
                    'pandas==0.23.3',
                    'numpy>=1.14.5'], #external packages as dependencies
)

print('packages', find_packages(include=['.','*'])),