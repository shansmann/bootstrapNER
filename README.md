# bootstrapNER

This project aims at bootstrapping NER for novel entities using distant supervision. Training datasets will be created 
using the Microsoft concept graph in a weakly supervised manner. 

NER architecture is based on the work by Ma and Hovy 
"End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF" as well as Lample et al. "Neural Architectures 
for Named Entity Recognition". 

# Results

| Experiment (15)         | Pretraining (5)  | Noise         | Prec   | Rec    | F1     |
| ----------------------- | ---------------- | ------------- | ------ | ------ | ------ |
| product_corpus_man      | False            | False         | 1      | 1      | 1      |
| sensor_corpus_sample_10 | False            | False         | 1      | 1      | 1      |
| sensor_corpus_sample_10 | True             | Dropout       | 1      | 1      | 1      |
| sensor_corpus_sample_10 | True             | Trace         | 1      | 1      | 1      |
| sensor_corpus_sample_10 | True             | Fix           | 1      | 1      | 1      |
