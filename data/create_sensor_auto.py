import time

from datastores import AvroCollection, Processor


path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/sensor_corpus/2.avro'
opath = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/sensor_corpus/train2.txt'
entities = ['product', 'organization']

product = AvroCollection(mode='auto', verbose=False, entities=entities)
start = time.time()
product.parse_text_data(path)
print('finished parsing text data. {}m elapsed'.format(int((time.time()-start)/60)))
start = time.time()
product.parse_annotation_data(path)
print('finished parsing annotation data. {}m elapsed'.format(int((time.time()-start)/60)))
print('tagged entities (token basis) with pronouns:', product.anno_counts_total)
print('pronoun matches:', product.pronoun_matches)
print('pronoun tokens lost:', product.pronoun_tokens_lost)
print('tagged entities (token basis) without pronouns:', product.anno_counts)

processor = Processor(product, verbose=False)
start = time.time()
processor.annotate_documents()
processor.write_conll(opath)

print('created conll data. {}m elapsed'.format(int((time.time()-start)/60)))
print('written entity count (token basis):', processor.anno_counts)
