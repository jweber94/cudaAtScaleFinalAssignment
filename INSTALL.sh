#!/usr/bin/bash

echo "Extracting data"
tar -xzf ./data/misc.tar.gz -C ./data/
echo "Converting data"
pip install -r helpers/requirements.txt
chmod a+x ./helpers/convertToPgm.py
./helpers/convertToPgm.py ./data/misc
echo "Installation of the data setup finished"