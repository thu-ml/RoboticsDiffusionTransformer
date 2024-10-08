#!/bin/bash

echo "Downloading CALVIN dataset..."

# Create calvin folder in ../datasets/calvin/
mkdir -p ../datasets/calvin/

cd ../datasets/calvin/

# You can use this for faster downloading
# aria2c -x 16 -s 16 http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip

wget http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip

echo "Unzipping CALVIN dataset..."

unzip task_ABC_D.zip

echo "Done downloading and unzipping CALVIN dataset."
