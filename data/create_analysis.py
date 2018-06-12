import time

from datastores import AvroCollection, Processor

path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/product_corpus_auto/train.avro'
opath = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/sensor_corpus/analysis.txt'

sensor = AvroCollection(name='product', mode='auto', verbose=False)
start = time.time()
sensor.parse_text_data(path)
print('finished parsing text data. {}m elapsed'.format(int((time.time() - start) / 60)))
start = time.time()
sensor.parse_annotation_data(path)
print('finished parsing annotation data. {}m elapsed'.format(int((time.time() - start) / 60)))

print('automatic tagged entities:', sensor.anno_counts)

processor = Processor(sensor, entities=['product', 'organization'])
start = time.time()
processor.create_conll_format(opath)
print('created conll data. {}m elapsed'.format(int((time.time() - start) / 60)))

print('overlaps:', processor.entity_overlaps)
print('written entity count:', processor.anno_counts)
