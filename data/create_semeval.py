from datastores import TextCollection, Processor


# train
train = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/semeval/original/train/'
otrain = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/semeval/original/train/0_train.txt'
semeval = TextCollection('semeval')
semeval.parse_text_data(train)
semeval.parse_annotation_data(train)
processor = Processor(semeval)
processor.create_conll_format(otrain)

# dev
dev = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/semeval/original/dev/'
odev = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/semeval/original/dev/0_dev.txt'
semeval = TextCollection('semeval')
semeval.parse_text_data(dev)
semeval.parse_annotation_data(dev)
processor = Processor(semeval)
processor.create_conll_format(odev)

# test
test = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/semeval/original/test/'
otest = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/semeval/original/test/0_test.txt'
semeval = TextCollection('semeval')
semeval.parse_text_data(test)
semeval.parse_annotation_data(test)
processor = Processor(semeval)
processor.create_conll_format(otest)
