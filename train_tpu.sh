#!/bin/bash
source "$HOME/bin/activate-tf1"

set -e

export TPU_NAME=grpc://0.tcp.ngrok.io:15992
export NOISY=1
export DEBUG=1

export TPU_NAME=grpc://0.tcp.ngrok.io:17042
export RESUME_PKL=./results/00035-stylegan2-animefaces-1gpu-config-a/network-snapshot-000398.pkl
export RESUME_KIMG=431.1
export RESUME_TIME=23613 # 6h33m33s

export TPU_NAME=grpc://0.tcp.ngrok.io:18248
export RESUME_PKL=./results/00039-stylegan2-animefaces-1gpu-config-a/network-snapshot-000527.pkl
export RESUME_KIMG=575.5
export RESUME_TIME=`math 8*3600 + 13*60 + 49` # 8h 13m 49s

export TPU_NAME=grpc://0.tcp.ngrok.io:15328
export RESUME_PKL=./results/00041-stylegan2-animefaces-1gpu-config-a/network-snapshot-000716.pkl
export RESUME_KIMG=716.5
export RESUME_TIME=`math 10*3600 + 13*60 + 49` # 10h 13m 49s

#config="config-f" # StyleGAN 2
config="config-a" # StyleGAN 1

data_dir=gs://sgappa-multi/stylegan-encoder/datasets
dataset=animefaces
mirror=false
metrics=none

set -x
exec python3 -m pdb -c continue run_training.py --data-dir "${data_dir}" --config="${config}" --dataset="${dataset}" --mirror-augment="${mirror}" --metrics="${metrics}" "$@"
