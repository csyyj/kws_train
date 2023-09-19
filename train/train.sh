#!/usr/bin/env bash
source activate pytorch
export PYTHONPATH=`pwd`/../
python train_script.py
