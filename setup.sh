#!/usr/bin/env bash

#create virtual environment
python3 -m venv env

# activate the virtual environment
source ./env/bin/activate

# install requirements
python3 -m pip install -r requirements.txt

# turn the virtual environment "off"
deactivate