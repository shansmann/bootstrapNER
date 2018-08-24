# bootstrapNER

This project aims at bootstrapping NER for novel entities using distant supervision. Training datasets will be created 
using the Microsoft concept graph in a weakly supervised manner. 

NER architecture is based on the work by Ma and Hovy 
"End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF" as well as Lample et al. "Neural Architectures 
for Named Entity Recognition". 

# Results

## Product Corpus / Sensor Corpus

| Experiment (20)         | Pretraining (5)  | Noise         | Silver Train F1   | Silver Dev F1    | Gold Test F1     |
| ----------------------- | ---------------- | ------------- | ----------------- | ---------------- | ---------------- |
| sensor_corpus_sample_10 | False            | False         | 1      | 1      | 1      |
| sensor_corpus_sample_10 | True             | Dropout       | 1      | 1      | 1      |
| sensor_corpus_sample_10 | True             | Trace         | 1      | 1      | 1      |
| sensor_corpus_sample_10 | True             | Fix Train     | 1      | 1      | 1      |
| sensor_corpus_sample_10 | True             | Fix Fix       | 1      | 1      | 1      |
| ----------------------- | ---------------- | ------------- | ------ | ------ | ------ |
| sensor_corpus_sample_20 | False            | False         | 1      | 1      | 1      |
| sensor_corpus_sample_20 | True             | Dropout       | 1      | 1      | 1      |
| sensor_corpus_sample_20 | True             | Trace         | 1      | 1      | 1      |
| sensor_corpus_sample_20 | True             | Fix Train     | 1      | 1      | 1      |
| sensor_corpus_sample_20 | True             | Fix Fix       | 1      | 1      | 1      |


## Semeval Corpus / Science Corpus

| Experiment (20)         | Pretraining (5)  | Noise         | Silver Train F1   | Silver Dev F1    | Gold Test F1     |
| ----------------------- | ---------------- | ------------- | ----------------- | ---------------- | ---------------- |
| science_corpus   | False            | False         | 0.8964      | 0.8861     | 0.1970      |
| science_corpus   | True             | Dropout       | 0.8844      | 0.8768      | 0.2186      |
| science_corpus   | True             | Trace         | 0.8920      | 0.8802      | 0.2045      |
| science_corpus   | True             | Fix Train     | 0.8858      | 0.8742      | 0.2041      |
| science_corpus   | True             | Fix Fix       | 1      | 1      | 1      |
