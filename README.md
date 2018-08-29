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
| sensor_corpus_sample_10 | False            | False         | 0.8042      | 0.8037      | 0.2066      |
| sensor_corpus_sample_10 | True             | Dropout       | 0.7762      | 0.7922      | 0.2625      |
| sensor_corpus_sample_10 | True             | Trace         | 0.7879      | 0.7938      | 0.2065      |
| sensor_corpus_sample_10 | True             | Fix Train     | 0.7813      | 0.7832      | 0.2104      |
| sensor_corpus_sample_10 | True             | Fix Fix       | 0.7089      | 0.7221      | 0.2563      |
| ----------------------- | ---------------- | ------------- | ------ | ------ | ------ |
| sensor_corpus_sample_20 | False            | False         | 0.8125      | 0.8006      | 0.2136      |
| sensor_corpus_sample_20 | True             | Dropout       | 0.7981      | 0.7853      | 0.1450      |
| sensor_corpus_sample_20 | True             | Trace         | 0.8017      | 0.7935      | 0.2028      |
| sensor_corpus_sample_20 | True             | Fix Train     | 0.7979      | 0.7889      | 0.2112      |
| sensor_corpus_sample_20 | True             | Fix Fix       | 0.8183      | 0.8075      | 0.2672      |


## Semeval Corpus / Science Corpus

| Experiment (20)         | Pretraining (5)  | Noise         | Silver Train F1   | Silver Dev F1    | Gold Test F1     |
| ----------------------- | ---------------- | ------------- | ----------------- | ---------------- | ---------------- |
| science_corpus   | False            | False         | 0.8964      | 0.8861     | 0.1970      |
| science_corpus   | True             | Dropout       | 0.8844      | 0.8768      | 0.2186      |
| science_corpus   | True             | Trace         | 0.8920      | 0.8802      | 0.2045      |
| science_corpus   | True             | Fix Train     | 0.8858      | 0.8742      | 0.2041      |
| science_corpus   | True             | Fix Fix       | 1      | 1      | 1      |
