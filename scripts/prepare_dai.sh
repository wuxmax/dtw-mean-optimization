#!/usr/bin/env bash

wget -P datasets https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip
unzip -P someone -d datasets datasets/UCRArchive_2018.zip

python3 -m venv venv
source venv/bin/activate

cd app
python3 -m pip install -r requirements.txt
