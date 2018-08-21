#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 python ner/Train_NER_Product.py > ner/logs/product/log_product.txt
CUDA_VISIBLE_DEVICES=4 python ner/Train_NER_Sensor_Sample.py False > ner/logs/product/log_sensor_10_false.txt
CUDA_VISIBLE_DEVICES=4 python ner/Train_NER_Sensor_Sample.py dropout > ner/logs/product/log_sensor_10_dropout.txt
CUDA_VISIBLE_DEVICES=4 python ner/Train_NER_Sensor_Sample.py trace > ner/logs/product/log_sensor_10_trace.txt
CUDA_VISIBLE_DEVICES=4 python ner/Train_NER_Sensor_Sample.py fix_train /mnt/hdd/experiments/shansmann/bootstrapNER/data/weights_product.txt > ner/logs/product/log_sensor_10_fix_train.txt
CUDA_VISIBLE_DEVICES=4 python ner/Train_NER_Sensor_Sample.py fix_fix /mnt/hdd/experiments/shansmann/bootstrapNER/data/weights_product.txt > ner/logs/product/log_sensor_10_fix_fix.txt
