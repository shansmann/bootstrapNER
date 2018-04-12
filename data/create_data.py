from objects import Word, Document, DataCollection, AvroCollection
from processor import Processor, HeuristikProcessor, AnnProcessor, AvroProcessor, XMLProcessor
from extract_concepts import ConceptExtractor

"""
path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/semeval/original/train/'

semeval = DataCollection('semeval')
semeval.parse_input_data(path)

semeval_processor_ann = AnnProcessor(semeval, path)
#semeval_processor_ann.create_conll_format('/Users/sebastianhansmann/Documents/Code/TU/mt/data/semeval/test/conll.tsv')
unique_annos = semeval_processor_ann.return_unique_annos("Process")
for anno in unique_annos:
	print(anno)
"""
"""
path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/sensor/sample_small.avro'
out_path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/sensor/sample_small_conll.tsv'
full_path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/sensor/test/test.avro'

avro = AvroCollection('avrotest')
avro.parse_input_data(path)

#avro_processor = AvroProcessor(avro)
#avro_processor.create_conll_format(out_path)
"""

path = 'uber_vico/original/texts/'
xml_path = 'uber_vico/original/'
out_path = 'uber_vico/original/conll/conll.tsv'

uber_vico = DataCollection('ubervico')
uber_vico.parse_input_data(path)

uber_vico_processor = XMLProcessor(uber_vico, xml_path)

unique_annos = uber_vico_processor.return_unique_annos("organization")
for anno in unique_annos:
	print(anno)
#uber_vico_processor.create_conll_format(out_path)
