from datastores import AvroCollection, Processor


path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/sensor/self_labeled/1.avro'
opath = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/sensor/self_labeled/1.txt'

sensor = AvroCollection('sensor')
sensor.parse_text_data(path)
sensor.parse_annotation_data(path)
processor = Processor(sensor, entities=['boot_ner_process', 'boot_ner_product', 'boot_ner_organization'])
processor.create_conll_format(opath)
