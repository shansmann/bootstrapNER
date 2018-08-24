#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=5 python ner/Train_NER_Sensor_Sample.py False > ner/logs/log_sensor_10_False_dirk_dropout.txt
CUDA_VISIBLE_DEVICES=5 python ner/Train_NER_Sensor_Sample.py dropout > ner/logs/log_sensor_10_dropout_dirk_dropout.txt