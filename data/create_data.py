from objects import Word, Document, DataCollection, AvroCollection
from processor import Processor, HeuristikProcessor, AnnProcessor, AvroProcessor

"""
path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/semeval/test/'

semeval = DataCollection('semeval')
semeval.parse_input_data(path)

semeval_processor_ann = AnnProcessor(semeval, path)
semeval_processor_ann.create_conll_format('/Users/sebastianhansmann/Documents/Code/TU/mt/data/semeval/test/conll.tsv')
"""

path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/sensor/sample_small.avro'
out_path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/sensor/sample_small_conll.tsv'

avro = AvroCollection('avrotest')
avro.parse_input_data(path)

avro_processor = AvroProcessor(avro)
avro_processor.create_conll_format(out_path)