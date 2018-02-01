from objects import Word, Document, DataCollection
from processor import Processor, HeuristikProcessor, AnnProcessor

path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/semeval/test/'

semeval = DataCollection('semeval')
semeval.parse_input_data(path)

#semeval_processor = Processor(semeval)
#semeval_processor.create_conll_format()


semeval_processor_ann = AnnProcessor(semeval, path)
semeval_processor_ann.create_conll_format('/Users/sebastianhansmann/Documents/Code/TU/mt/data/semeval/test/conll.tsv')