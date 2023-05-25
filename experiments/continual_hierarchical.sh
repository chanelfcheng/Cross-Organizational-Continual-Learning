#!/bin/bash


cd ..

# TODO: change --dataset-paths to the correct path where your data is stored
python continual_hierarchical.py \
--exp-name $(basename ${0%".sh"}) \
--dataset-names "cic-2018,usb-2021" \
--dataset-paths "../data/CIC-IDS2018/DoS+BruteForce,../data/USB-IDS2021" \
--dataset-classes "benign,hulk,slowloris,slowhttp,goldeneye,tcpflood,ftp,ssh" \
--rename-binary "Benign,Attack,Attack,Attack,Attack,Attack,Attack,Attack" \
--rename-super ",DoS,DoS,DoS,DoS,DoS,BruteForce,BruteForce" \
--rename-sub ",DoS-Hulk,DoS-Slowloris,DoS-SlowHttpTest,DoS-GoldenEye,DoS-TCPFlood,BruteForce-FTP,BruteForce-SSH" \
--arch "mlp"