#!/usr/bin/env bash

#wget http://nlp.stanford.edu/data/glove.840B.300d.zip --directory-prefix=./ner/
#unzip glove.840B.300d.zip
#rm glove.840B.300d.zip

wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/product_corpus_man/train.txt' --directory-prefix=./data/product_corpus_man/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/product_corpus_man/dev.txt' --directory-prefix=./data/product_corpus_man/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/product_corpus_man/test.txt' --directory-prefix=./data/product_corpus_man/

wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/sensor_sample_10/train.txt' --directory-prefix=./data/sensor_sample_10/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/sensor_sample_10/dev.txt' --directory-prefix=./data/sensor_sample_10/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/sensor_sample_10/test.txt' --directory-prefix=./data/sensor_sample_10/

wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/sensor_sample_20/train.txt' --directory-prefix=./data/sensor_sample_20/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/sensor_sample_20/dev.txt' --directory-prefix=./data/sensor_sample_20/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/sensor_sample_20/test.txt' --directory-prefix=./data/sensor_sample_20/

wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/sensor_sample_30/train.txt' --directory-prefix=./data/sensor_sample_30/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/sensor_sample_30/dev.txt' --directory-prefix=./data/sensor_sample_30/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/sensor_sample_30/test.txt' --directory-prefix=./data/sensor_sample_30/

wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/semeval_man/train.txt' --directory-prefix=./data/semeval_man/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/semeval_man/dev.txt' --directory-prefix=./data/semeval_man/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/semeval_man/test.txt' --directory-prefix=./data/semeval_man/

wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/science_corpus/train.txt' --directory-prefix=./data/science_corpus/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/science_corpus/dev.txt' --directory-prefix=./data/science_corpus/
wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/bootstrap-ner-data/science_corpus/test.txt' --directory-prefix=./data/science_corpus/

#wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/conll/train.txt' --directory-prefix=./data/conll/
#wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/conll/dev.txt' --directory-prefix=./data/conll/
#wget --no-check-certificate --no-proxy 'https://s3-eu-west-1.amazonaws.com/conll/test.txt' --directory-prefix=./data/conll/