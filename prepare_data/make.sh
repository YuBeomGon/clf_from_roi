#!/bin/bash
seed=2

python make_csv.py --seed $seed
python make_json.py --seed $seed
python make_clf_csv.py --seed $seed

cp saved/*.csv ../lbp_data/
cp saved/*.json ../lbp_data/