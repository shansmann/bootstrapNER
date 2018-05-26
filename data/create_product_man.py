from datastores import AvroCollection, Processor


path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/product_corpus/man_labeled/train.avro'
opath = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/product_corpus/man_labeled/train.txt'

product = AvroCollection('product')
product.parse_text_data(path)
product.parse_annotation_data(path)
processor = Processor(product, entities=['product', 'organization'])
processor.create_conll_format(opath)
