import time

from datastores import TextCollection, Processor
from concept_extractor import ConceptNet, ConceptExtractor

test = ConceptNet(verbose=True, create=False)

surfaces = test.query_concept_net_for_concept('company')
print(len(surfaces))
