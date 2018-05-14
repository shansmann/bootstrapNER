import glob
import os
from avro.datafile import DataFileReader
from avro.io import DatumReader
from os.path import basename
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
import logging, coloredlogs
import hashlib
import itertools

from .config import *

coloredlogs.install()


class Token:
	def __init__(self, word, start, end, entity=OTHER_ENTITY, score=float(-1)):
		self.word = word
		self.start = int(start)
		self.end = int(end)
		self.entity = entity
		self.score = score


class Document:
	def __init__(self, id, tokens, sentence_breaks=None, plain_text=None):
		self.id = id
		self.tokens = tokens
		self.sentence_breaks = sentence_breaks
		self.plain = plain_text


class Collection:
	def __init__(self, name):
		self.name = name
		self.documents = []
		self.annotations = []

	def parse_text_data(self):
		pass

	def parse_annotation_data(self):
		pass

	def clean_annotations(self, verbose=False):
		# remove duplicates
		for annotatedDocument in self.annotations:
			new_annotations = []
			tokens = annotatedDocument.tokens
			for i in range(len(tokens)):
				discard_anno = False
				for j in range(i + 1, len(tokens)):
					discard_anno = self._compare_annotations(tokens[i], tokens[j])
					if discard_anno:
						break
				if not discard_anno:
					new_annotations.append(tokens[i])
			annotatedDocument.tokens = new_annotations
			if verbose:
				[print(x.word, x.entity) for x in new_annotations]

	def _compare_annotations(self, anno1, anno2):
		"""
		returns True if annotation is contained in other
		:param anno1:
		:param anno2:
		:return Boolean:
		"""
		if anno1.entity != anno2.entity:
			return False
		range_anno = range(anno2.start, anno2.end)
		if anno1.start in range_anno and anno1.end-1 in range_anno:
			# entity contained in other entity
			logging.info('merging annotations:')
			logging.info('word1: {}, range: {} with word2: {}, range: {}'.format(anno1.word,
																				 range(anno1.start, anno1.end),
																				 anno2.word,
																				 range_anno))
			return True
		else:
			# TODO: intersect not handled
			return False


class TextCollection(Collection):
	def parse_text_data(self, path):
		docs = []
		for filename in glob.glob(os.path.join(path, '*.txt')):
			file = open(filename, 'r')
			base_name = basename(filename).split('.')[0]
			file_content = file.read()
			words = []
			sentence_breaks = []
			for token in self._spans(file_content):
				words.append(Token(token[0], token[1], token[2]))
			for start, end in PunktSentenceTokenizer().span_tokenize(file_content):
				sentence_breaks.append(end)
			doc = Document(base_name, words, sentence_breaks, file_content)
			docs.append(doc)
			file.close()
			if len(docs) % 5 == 0:
				logging.info('finished {} documents.'.format(len(docs)))
				break
		self.documents = docs

	def parse_annotation_data(self, path):
		annotated_docs = []
		for filename in glob.glob(os.path.join(path, '*.ann')):
			f_anno = open(filename, 'r')
			base_name = basename(filename).split('.')[0]
			annotations = []
			for line in f_anno:
				anno_inst = line.strip('\n').split('\t')
				if len(anno_inst) == 3:
					anno_inst1 = anno_inst[1].split(' ')
					if len(anno_inst1) == 3:
						keytype, start, end = anno_inst1
					else:
						keytype, start, _, end = anno_inst1
					if not keytype.endswith('-of'):
						surface = anno_inst[2]
						annotations.append(Token(surface, int(start), int(end), keytype, float(1)))
			annotated_doc = Document(base_name, annotations)
			annotated_docs.append(annotated_doc)
			f_anno.close()
			if len(annotated_docs) % 5 == 0:
				logging.info('finished {} documents.'.format(len(annotated_docs)))
				break
		self.annotations = annotated_docs

	def _spans(self, txt):
		tokens = word_tokenize(txt)
		offset = 0
		for token in tokens:
			offset = txt.find(token, offset)
			yield token, offset, offset + len(token)
			offset += len(token)


class AvroCollection(Collection):
	def parse_text_data(self, path):
		docs = []
		reader = DataFileReader(open(path, 'rb'), DatumReader())
		for document in reader:
			file_content = document['text']
			m = hashlib.md5()
			m.update(document['id'].encode('utf-8'))
			idd = str(int(m.hexdigest(), 16))[0:12]
			words = []
			sentence_breaks = []
			for token in document['tokens']:
				start = token['span']['start']
				end = token['span']['end']
				#TODO: handle LEMMA
				word = file_content[start:end]
				words.append(Token(word, start, end))
			for sentence in document['sentences']:
				end = sentence['span']['end']
				sentence_breaks.append(end)
			doc = Document(idd, words, sentence_breaks, file_content)
			docs.append(doc)
			if len(docs) % 5 == 0:
				logging.info('finished {} documents.'.format(len(docs)))
				break
		reader.close()
		self.documents = docs

	def parse_annotation_data(self, path):
		annotated_docs = []
		reader = DataFileReader(open(path, 'rb'), DatumReader())
		for document in reader:
			m = hashlib.md5()
			m.update(document['id'].encode('utf-8'))
			idd = str(int(m.hexdigest(), 16))[0:12]
			annotations = []
			for token in document['conceptMentions']:
				# for now only sprout labeled tokens
				concept_meta = token.get('attributes')
				if concept_meta:
					word = token.get('normalizedValue')
					start = token.get('span').get('start')
					end = token.get('span').get('end')
					entity = concept_meta.get('sprout_ner_tag')
					score = concept_meta.get('ms_concept_graph_rep_e_c')
					tok = Token(word, start, end, entity, float(score))
					annotations.append(tok)
			annotated_doc = Document(idd, annotations)
			annotated_docs.append(annotated_doc)
			if len(annotated_docs) % 5 == 0:
				logging.info('finished {} documents.'.format(len(annotated_docs)))
				break
		reader.close()
		self.annotations = annotated_docs


class Processor:
	def __init__(self, Collection, entities=[]):
		self.text_collection = Collection.documents
		self.annotation_collection = Collection.annotations
		self.entities = entities
		self.entity_overlaps = 0
		self.index_errors = 0

	def _match_tokens(self, token, anno):
		prefix = ''
		if anno.entity in self.entities:
			if token.start >= anno.start and token.end <= anno.end:
				# check words
				if token.word in anno.word:
					# token match
					if token.score < anno.score:
						# IOB2 encoding
						anno_tokens = word_tokenize(anno.word)
						if len(anno_tokens) > 1:
							if anno.word.index(token.word) == 0:
								# first word
								prefix = 'B-'
							else:
								prefix = 'I-'
						else:
							prefix = 'B-'
						return prefix
					else:
						# token overlap
						logging.warning('token overlap. Word: {}, Previous: {}-{}, Suggestion: {}-{}'.format(token.word,
																									   token.entity,
																									   token.score,
																									   anno.entity,
																									   anno.score))
						self.entity_overlaps += 1
						return prefix
				else:
					# token not in annotation, index fail
					logging.warning('token not in annotation! Word:{} Annotation:{}'.format(token.word,
																							anno.word))
					self.index_errors += 1
					return prefix
			else:
				return prefix
		else:
			return prefix

	def create_conll_format(self, outpath):
		with open(outpath, 'w') as record_file:
			for document in self.text_collection:
				idd = document.id
				for token in document.tokens:
					# check annotated docs
					for annotated_doc in self.annotation_collection:
						if annotated_doc.id == idd:
							# right file
							for anno in annotated_doc.tokens:
								# check annotations
								match = self._match_tokens(token, anno)
								if match:
									token.score = anno.score
									token.entity = match + anno.entity
						else:
							continue
					if (token.start - 1) in (document.sentence_breaks):
						record_file.write('\n')
					line = '{}\t{}\t{}\t{}\n'.format(token.entity, token.start, token.end, token.word)
					record_file.write(line)
				record_file.write('\n')