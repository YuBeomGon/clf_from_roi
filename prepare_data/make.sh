#!/bin/bash

python make_csv.py
python make_json.py
python make_clf_csv.py

cp saved/*.csv ../lbp_data/
cp saved/*.json ../lbp_data/