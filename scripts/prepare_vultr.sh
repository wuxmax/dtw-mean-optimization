#!/usr/bin/env bash

apt update
apt install -y python3-pip 
apt install -y python3-venv
apt install -y unzip

mkdir results
mkdir datasets

wget -P datasets https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip
unzip -P someone -d datasets/UCRArchive_2018 datasets/UCRArchive_2018.zip

python3 -m venv venv
source venv/bin/activate

cd app
python3 -m pip install -r requirements.txt
