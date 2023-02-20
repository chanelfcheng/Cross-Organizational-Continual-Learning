#!/bin/bash


cd ..

# TODO: change --dataset-paths to the correct path where your data is stored
python continual.py \
--exp-name $(basename ${0%".sh"}) \
--dataset-names "cic-2018,usb-2021" \
--dataset-paths "../data/CIC-IDS2018/DoS+Hulk,../data/USB-IDS2021" \
--dataset-classes "benign,tcpflood,ssh" \
--rename-labels "Benign,DoS,BruteForce" \
--arch "mlp"