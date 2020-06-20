apt install pyhton3-pip
pyhton3 -m venv venv
source venv/bin/activate

mkdir results
mkdir datasets

wget https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip
unzip -P someone UCRArchive_2018.zip

