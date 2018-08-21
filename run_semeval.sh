#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=5 python ner/Train_NER_Semeval.py > ner/logs/semeval/log_semeval.txt
CUDA_VISIBLE_DEVICES=5 python ner/Train_NER_Science.py False > ner/logs/semeval/log_science_false.txt
CUDA_VISIBLE_DEVICES=5 python ner/Train_NER_Science.py dropout > ner/logs/semeval/log_science_dropout.txt
CUDA_VISIBLE_DEVICES=5 python ner/Train_NER_Science.py trace > ner/logs/semeval/log_science_trace.txt
CUDA_VISIBLE_DEVICES=5 python ner/Train_NER_Science.py fix_train /mnt/hdd/experiments/shansmann/bootstrapNER/data/weights_sem.txt > ner/logs/semeval/log_science_fix_train.txt
CUDA_VISIBLE_DEVICES=5 python ner/Train_NER_Science.py fix_fix /mnt/hdd/experiments/shansmann/bootstrapNER/data/weights_sem.txt > ner/logs/semeval/log_science_fix_fix.txt


#!/usr/bin/env bash

#python ner/Train_NER_Semeval.py > log_semeval.txt
#python ner/Train_NER_Science.py False > log_science_false.txt
#python ner/Train_NER_Science.py dropout > log_science_dropout.txt
#python ner/Train_NER_Science.py trace > log_science_trace.txt
#python ner/Train_NER_Science.py fix_train /Users/sebastianhansmann/Documents/Code/TU/mt/data/weights_sem.txt > log_science_fix_train.txt
#python ner/Train_NER_Science.py fix_fix /Users/sebastianhansmann/Documents/Code/TU/mt/data/weights_sem.txt > log_science_fix_fix.txt
