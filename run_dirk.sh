#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=5 python ner/Train_NER_Sensor_Sample.py False > ner/logs/log_sensor_10_False_dirk_dropout.txt
#CUDA_VISIBLE_DEVICES=5 python ner/Train_NER_Sensor_Sample.py dropout > ner/logs/log_sensor_10_dropout_dirk_dropout.txt

CUDA_VISIBLE_DEVICES=5 python ner/Train_NER_Science.py False > ner/logs/log_science_False_dirk_token.txt
CUDA_VISIBLE_DEVICES=5 python ner/Train_NER_Science.py dropout > ner/logs/log_science_dropout_dirk_token.txt