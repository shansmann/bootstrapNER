import time

from datastores import AvroCollection, Processor


path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/product_corpus_man/test.avro'
opath = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/product_corpus_man/hansi.txt'

sensor = AvroCollection(name='product', mode='man', verbose=True)
start = time.time()
sensor.parse_text_data(path)
print('finished parsing text data. {}m elapsed'.format(int((time.time()-start)/60)))
start = time.time()
sensor.parse_annotation_data(path)
print('finished parsing annotation data. {}m elapsed'.format(int((time.time()-start)/60)))
processor = Processor(sensor, entities=['product', 'organization'], verbose=True)
start = time.time()
processor.create_conll_format(opath)
print('created conll data. {}m elapsed'.format(int((time.time()-start)/60)))
