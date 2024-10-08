# Download the dataset to ../datasets/bridgev2
mkdir -p ../datasets/bridgev2
wget -O ../datasets/bridgev2/demos_8_17.zip https://rail.eecs.berkeley.edu/datasets/bridge_release/data/demos_8_17.zip
mkdir -p ../datasets/bridgev2/raw
# Unzip the dataset
unzip '../datasets/bridgev2/*.zip' -d ../datasets/bridgev2/raw
# Convert the dataset to numpy
python bridgedata_raw_to_numpy.py --input ../datasets/bridgev2/raw --output ../datasets/bridgev2/npy
# Convert the dataset to tfrecords
python bridgedata_numpy_to_tfrecord.py --input ../datasets/bridgev2/npy --output ../datasets/bridgev2/tfrecords
# Remove the raw data and numpy data
rm -rf ../datasets/bridgev2/raw
rm -rf ../datasets/bridgev2/npy