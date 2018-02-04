import glob
import os
from os.path import basename
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
import config

class Word:
	def __init__(self, word, start, end, entity = config.OTHER_ENTITY):
		self.word = word
		self.start = int(start)
		self.end = int(end)
		self.entity = entity

	def __repr__(self):
		return str(self.__dict__)


class Annotation:
	def __init__(self, surface, start, end, entity):
		self.word = surface
		self.start = start
		self.end = end
		self.entity = entity

	def __repr__(self):
		return str(self.__dict__)


class Document:
	def __init__(self, id, sentence_breaks, tokens, plain_text):
		self.id = id
		self.sentence_breaks = sentence_breaks
		self.tokens = tokens
		self.plain = plain_text


class DataCollection:
	def __init__(self, name):
		self.name = name
		self.documents = []

	def _spans(self, txt):
		tokens = word_tokenize(txt)
		offset = 0
		for token in tokens:
			offset = txt.find(token, offset)
			yield token, offset, offset + len(token)
			offset += len(token)

	def parse_input_data(self, path):
		docs = []
		for filename in glob.glob(os.path.join(path, '*.txt')):
			file = open(filename, "r")
			base_name = basename(filename).split('.')[0]
			file_content = file.read()
			words = []
			sentence_breaks = []
			for token in self._spans(file_content):
				words.append(Word(token[0], token[1], token[2]))
			for start, end in PunktSentenceTokenizer().span_tokenize(file_content):
				sentence_breaks.append(end)
			doc = Document(base_name, sentence_breaks, words, file_content)
			docs.append(doc)

			file.close()

		self.documents = docs