import time

from datastores import AvroCollection, Processor
from concept_extractor import ConceptNet, ConceptExtractor

path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/product_corpus_man/full.avro'
opath = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/product_corpus_man/full2.txt'
entities = ['product', 'organization']

product = AvroCollection(mode='man', verbose=False, entities=entities)
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

#concept_net = ConceptNet(verbose=True)

with open('additional_org.txt') as f:
	add_surfaces = [x.strip().lower() for x in f.readlines()]

"""
for entity in entities:
	#path = 'concepts/concepts_{}_filtered.txt'.format(entities[0])
	concepts = ConceptExtractor(collection=product,
								concept_net=concept_net,
								entity=entities[0],
								verbose=True,
								additional_surfaces=add_surfaces,
								top_n_concepts=1000,)
								#load_concepts=path)

	concepts.query_concepts()
	concepts.query_surfaces()
	concepts.get_statistics()
	concepts.write_files()
	break
"""
processor = Processor(product, verbose=False)
start = time.time()
processor.annotate_documents()
processor.write_conll(opath)

print('created conll data. {}m elapsed'.format(int((time.time()-start)/60)))
print('written entity count (token basis):', processor.anno_counts)
