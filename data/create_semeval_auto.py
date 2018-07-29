import time

from datastores import TextCollection, Processor, AvroCollection
from concept_extractor import ConceptNet, ConceptExtractor

path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/semeval_auto/train.avro'
opath = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/semeval_auto/train.txt'
entities = ['process', 'material', 'task']

semeval = AvroCollection(mode='auto', verbose=False, entities=entities)
#semeval = TextCollection(verbose=False, entities=entities)
start = time.time()
semeval.parse_text_data(path)
print('finished parsing text data. {}m elapsed'.format(int((time.time()-start)/60)))
start = time.time()
semeval.parse_annotation_data(path)
print('finished parsing annotation data. {}m elapsed'.format(int((time.time()-start)/60)))
print('tagged entities (token basis):', semeval.anno_counts)

"""
for entity in entities:
	#path = 'concepts/concepts_{}_filtered.txt'.format(entities[2])
	concepts = ConceptExtractor(collection=semeval,
								concept_net=ConceptNet(verbose=True),
								entity=entities[2],
								verbose=True,
								top_n_concepts=1000,)
								#load_concepts=path,)
	#concepts.query_concepts()
	#concepts.query_surfaces()
	#concepts.get_statistics()
	#concepts.write_files()
	break
"""
processor = Processor(semeval, verbose=False)
start = time.time()
processor.annotate_documents()
processor.write_conll(opath)

print('created conll data. {}m elapsed'.format(int((time.time()-start)/60)))
print('written entity count (token basis):', processor.anno_counts)
