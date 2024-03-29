#!/bin/bash


cd ..

# NOTE: this experiment does not run currently due to not being able to one-hot
# encode the protocols (missing benign class).
# TODO: change --dataset-paths to the correct path where your data is stored
python continual.py \
--exp-name $(basename ${0%".sh"}) \
--dataset-names "cic-2018,usb-2021" \
--dataset-paths "../data/CIC-IDS2018/Hulk+FTP,../data/USB-IDS2021" \
--dataset-classes "hulk,ftp" \
--rename-labels "DoS,BruteForce" \
--arch "mlp"