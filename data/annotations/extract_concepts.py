import json
from collections import Counter

filename = 'material_concepts.json'
oname = 'f_min_material_concepts.json'


def get_all_concepts(data):
	tmp = []
	for datum in data["data"]:
		if datum["concepts"]:
			for concept in datum["concepts"].keys():
				tmp.append(concept)
	return tmp

def get_concept_counter(concepts):
	return Counter(concepts)

def get_all_products_with_concepts(data):
	tmp = []
	for datum in data["data"]:
		if datum["concepts"]:
			tmp.append(datum)
	return tmp

def match_concept(concept, concepts):
	if concept in concepts.keys():
		return True
	else:
		return False

def filter_concepts(concepts, filters):
	tmp = []
	for concept in concepts:
		if concept not in filters:
			tmp.append(concept)
	return tmp

def get_minimum_concepts(counter, data):
	unfound_products = get_all_products_with_concepts(data)
	concepts = []

	for (concept, count) in counter.most_common():

		before_filtering = len(unfound_products)
		unfound_products = [x for x in unfound_products if
							not match_concept(concept, x["concepts"])]

		if not unfound_products:
			break

		after_filtering = len(unfound_products)

		if before_filtering > after_filtering:
			concepts.append(concept)

	return concepts

def write_concepts(oname, concepts):
	with open(oname, 'w') as outfile:
		json.dump({
			"concepts": concepts
		}, outfile)


if __name__ == '__main__':
	# reading
	data = json.load(open(filename))

	# extracting concepts
	concepts = get_all_concepts(data)
	# concept occurences
	cnt = get_concept_counter(concepts)

	# get min concepts
	concepts = get_minimum_concepts(cnt, data)

	# filter concepts
	filter_semeval = ["folder", "term", "word", "letter", "topic", "property", "subject", "medium", "successfulresultswerealsoachievedbynabhaniwithultrahard cutting tool material", "complex shape", "transgender family organization", "image", "document", "design", "thatsomesolidacids", "threedimensional object", "notion", "factor", "advanced topic", "god form", "concept", "aspect", "section heading", "method", "design aspect", "effect", "abrasive suspension", "technique", "process", "class", "case", "key word", "keyword"]
	filter_uber_vico = ["pronoun", "collective noun", "basic principle", "interrogative word", "topic",
						"ambiguous reference", "impersonal pronoun", "activity", "issue", "term", "unit",
						"religious symbol", "neutral", "spirit being", "cycle", "exit method", "popular category",
						"organization"]

	filtered = filter_concepts(concepts, filter_semeval)
	for concept in filtered:
		print(concept)

	write_concepts(oname, filter_concepts(concepts, filter_uber_vico))
