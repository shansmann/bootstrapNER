#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 python Train_NER_Product.py > log_product.txt
CUDA_VISIBLE_DEVICES=4 python Train_NER_Sensor_Sample.py False > log_sensor_10_false.txt
CUDA_VISIBLE_DEVICES=4 python Train_NER_Sensor_Sample.py dropout > log_sensor_10_dropout.txt
CUDA_VISIBLE_DEVICES=4 python Train_NER_Sensor_Sample.py trace > log_sensor_10_trace.txt
CUDA_VISIBLE_DEVICES=4 python Train_NER_Sensor_Sample.py fix_train /mnt/hdd/experiments/shansmann/bootstrapNER/data/weights_product.txt > log_sensor_10_fix_train.txt
CUDA_VISIBLE_DEVICES=4 python Train_NER_Sensor_Sample.py fix_fix /mnt/hdd/experiments/shansmann/bootstrapNER/data/weights_product.txt > log_sensor_10_fix_fix.txt
