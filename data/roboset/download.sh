#!/bin/bash

while true; do
    python download.py
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Download exited with code $EXIT_CODE. Restarting..."
    else
        echo "Download exited with code 0. Not restarting."
        break
    fi
done

# Unzip all the files in the ../datasets/roboset/ directory
cd ../datasets/roboset/
for file in *.tar.gz; do
    tar -xzvf "$file"
done

## Convert the dataset to tfrecords
python hdf5totfrecords.py