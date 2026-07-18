#!/usr/bin/bash

# create virtual env
python -m venv env
# activate env
source ./env/bin/activate
# install requirements
pip install --upgrade pip

pip install -r requirements.txt

#pip install -U pip setuptools wheel
#pip install -U spacy

# close the environment
deactivate