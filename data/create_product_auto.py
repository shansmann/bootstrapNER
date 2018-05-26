from datastores import AvroCollection, Processor


path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/product_corpus/auto_labeled/train.avro'
opath = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/product_corpus/auto_labeled/train.txt'

sensor = AvroCollection('product')
sensor.parse_text_data(path)
sensor.parse_annotation_data(path)
processor = Processor(sensor, entities=['product', 'organization'])
processor.create_conll_format(opath)
