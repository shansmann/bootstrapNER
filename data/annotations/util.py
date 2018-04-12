"""
utility functions for tokenizing, sentencing, parsing xml, etc
"""
import re
import xmltodict

from nltk.tokenize import sent_tokenize, word_tokenize


def remove_html(text):
	cleanr = re.compile('<.*?>')
	return re.sub(cleanr, '', text)

def extract_text_from_xml(fname):
	with open(fname) as fd:
		doc = xmltodict.parse(fd.read())
		return doc['Document']['TEXT']['#text']

def extract_entities_from_xml(fname):
	tmp = []
	with open(fname) as fd:
		doc = xmltodict.parse(fd.read())
		entities = doc['Document']['Entities']['Entity']
		for elem in entities:
			word = elem['@surface']
			if word:
				start = elem['@startPos']
				end = elem['@endPos']
				typ = elem['@type']
				tmp.append((word, (start, end), typ))
	return tmp

def validate_tagging(text, entities):
	for entity in entities:
		start = int(entity[1][0])
		end = int(entity[1][1])
		ent = text[start:end]
		if not entity[0] == ent:
			return False
	return True

def get_conll_from_xml(text, entities):
	sentences = [tokenize(x) for x in sentencing(text)]
	for sentence in sentences:
		for word in sentence:
			# partial matches?
			match = 'O'
			if word in [x[0] for x in entities]:
				for ent in entities:
					if ent[0] == word:
						match = ent[2]
			print(word + '\t' + match)
		print('\n')

def tokenize(sentence):
	return word_tokenize(sentence)

def sentencing(text):
	return sent_tokenize(text)

if __name__ == '__main__':
	fname = "elif_vicu_uber_edited_20180319/uber/1c6cf7a0127b8daa.xml"

	entities = extract_entities_from_xml(fname)
	text = extract_text_from_xml(fname)
	if validate_tagging(text, entities):
		get_conll_from_xml(text, entities)
