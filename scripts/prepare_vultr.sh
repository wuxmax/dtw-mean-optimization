apt install -y python3-pip 
apt install -y python3-pipunzip

mkdir results
mkdir datasets

wget https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip
unzip -P someone UCRArchive_2018.zip

pyhton3 -m venv venv
python3 -m pip install -r requirements.txt
source venv/bin/activate
