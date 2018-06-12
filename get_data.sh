#!/usr/bin/env bash

#wget http://nlp.stanford.edu/data/glove.840B.300d.zip --directory-prefix=./ner/
#unzip glove.840B.300d.zip
#rm glove.840B.300d.zip

wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/sensor-corpus-auto/train.txt' --directory-prefix=./data/sensor_corpus/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/sensor-corpus-auto/dev.txt' --directory-prefix=./data/sensor_corpus/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/sensor-corpus-auto/test.txt' --directory-prefix=./data/sensor_corpus/

wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/conll/train.txt' --directory-prefix=./data/conll/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/conll/dev.txt' --directory-prefix=./data/conll/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/conll/test.txt' --directory-prefix=./data/conll/