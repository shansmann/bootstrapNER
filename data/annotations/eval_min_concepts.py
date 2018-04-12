import glob
import os
import json


def get_duplicates(lists):
	seen = set()
	repeated = set()
	for l in lists:
		for i in set(l):
			if i in seen:
				repeated.add(i)
			else:
				seen.add(i)
	return list(repeated)

def read_files(folder, entities=["min"]):
	lists = []
	for filename in glob.glob(os.path.join(folder, 'f_min*.json')):
		if any([x in filename for x in entities]):
			print(filename)
			with open(filename) as data_file:
				data = json.load(data_file)
				lists.append(data["concepts"])
	return lists

def filter_concepts(concepts, filters):
	tmp = []
	for concept in concepts:
		if concept not in filters:
			tmp.append(concept)
	return tmp

if __name__ == '__main__':
	filter_semeval = ["folder", "term", "word", "letter", "topic", "property", "subject", "medium", "successfulresultswerealsoachievedbynabhaniwithultrahard cutting tool material", "complex shape", "transgender family organization", "image", "document", "design", "thatsomesolidacids", "threedimensional object", "notion", "factor", "advanced topic", "god form", "concept", "aspect", "section heading"]

	semeval = read_files('', ["material", "task", "process"])
	semeval_concepts = [item for sublist in semeval for item in sublist]
	#print(len(semeval_concepts))
	semeval_filtered = filter_concepts(semeval_concepts, filter_semeval)
	#print(len(semeval_filtered))
	#print(semeval_filtered)
	print(get_duplicates(read_files('', ["material", "task", "process"])))

	filter_uber_vico = ["pronoun", "collective noun", "basic principle", "interrogative word", "topic", "ambiguous reference", "impersonal pronoun", "activity", "issue", "term", "unit", "religious symbol", "neutral", "spirit being", "cycle", "exit method", "popular category", "organization"]
	#print(get_duplicates(read_files('', ["org", "prod"])))
	uber_vico = read_files('', ["prod", "org"])
	uber_vico_concepts = [item for sublist in uber_vico for item in sublist]
	#print(len(uber_vico_concepts))
	uber_vico_filtered = filter_concepts(uber_vico_concepts, filter_uber_vico)
	#print(len(uber_vico_filtered))
	#print(uber_vico_filtered)
