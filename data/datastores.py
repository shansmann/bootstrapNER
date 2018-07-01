import glob
import os
import re
from collections import defaultdict
from os.path import basename

import logging, coloredlogs
import hashlib
from avro.datafile import DataFileReader
from avro.io import DatumReader
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer

import config

coloredlogs.install()


class Token:
	def __init__(self, word, start, end, entity=config.OTHER_ENTITY, score=float(-1), plain_entity=config.OTHER_ENTITY):
		self.word = word
		self.start = int(start)
		self.end = int(end)
		self.entity = entity
		self.score = score
		self.plain_entity = plain_entity


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

	def parse_text_data(self, path):
		pass

	def parse_annotation_data(self, path):
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
	def __init__(self, verbose=False, entities=None):
		self.verbose = verbose
		self.entities = entities or []
		self.anno_counts = defaultdict(int)
		super(Collection, self).__init__()

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
			if len(docs) % 100 == 0:
				logging.info('finished {} documents.'.format(len(docs)))
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
						if surface and start and end and keytype and keytype in self.entities:
							self.anno_counts[keytype] += len(word_tokenize(surface))
							annotations.append(Token(surface, int(start), int(end), keytype, float(1)))
			annotated_doc = Document(base_name, annotations)
			annotated_docs.append(annotated_doc)
			f_anno.close()
			if len(annotated_docs) % 100 == 0:
				logging.info('finished {} documents.'.format(len(annotated_docs)))
		self.annotations = annotated_docs

	def _spans(self, txt):
		tokens = word_tokenize(txt)
		offset = 0
		for token in tokens:
			offset = txt.find(token, offset)
			yield token, offset, offset + len(token)
			offset += len(token)


class AvroCollection(Collection):
	def __init__(self, mode, verbose=False, entities=None):
		self.mode = mode
		self.verbose = verbose
		self.entities = entities or []
		self.anno_counts_total = defaultdict(int)
		self.anno_counts = defaultdict(int)
		self.pronoun_matches = defaultdict(int)
		self.pronoun_tokens_lost = defaultdict(int)
		super(Collection, self).__init__()

	def parse_text_data(self, path):
		docs = []
		reader = DataFileReader(open(path, 'rb'), DatumReader())
		#offset = 0
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
				if re.search(r"\s", word) or len(word) == 0:
					if self.verbose:
						logging.warning('token: {} contains spaces or is empty, skipping. document id: {}'.format(word, idd))
					continue
				words.append(Token(word, start, end))
			for sentence in document['sentences']:
				end = sentence['span']['end']
				sentence_breaks.append(end)
			doc = Document(idd, words, sentence_breaks, file_content)
			docs.append(doc)
			if len(docs) % 100 == 0:
				logging.info('finished {} documents.'.format(len(docs)))
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
				word = token.get('normalizedValue', None)
				start = token.get('span').get('start', None)
				end = token.get('span').get('end', None)
				if self.mode == 'auto':
					concept_meta = token.get('attributes')
					if concept_meta:
						entity = concept_meta.get('sprout_ner_tag', '').replace('boot_ner_', '')
						score = concept_meta.get('ms_concept_graph_rep_e_c', None)
						if word and start and end and entity and score and entity in self.entities:
							tok = Token(word, start, end, entity, float(score))
							self.anno_counts[entity] += len(word_tokenize(word))
							annotations.append(tok)
						elif self.verbose:
							logging.warning('invalid annotation found in document {}'.format(idd))
				else:
					forbidden_pattern = re.compile(r"(company|firm|business|organization|we|our|ours|us|it|its|I|me|my|mine|you|your|yours|they|them|theirs|their|she|her|hers|him|his|he|itself|ourselves|themselves|myself|yourself|yourselves|himself|herself|which|who|whom|whose|whichever|whoever|whomever|those|these|this. + | that. + | this | that)", re.I | re.U)
					entity = token.get('type', None)
					if word and start and end and entity and entity in self.entities:
						self.anno_counts_total[entity] += len(word_tokenize(word))
						if forbidden_pattern.match(word):
							self.pronoun_matches[entity] += 1
							self.pronoun_tokens_lost[entity] += len(word_tokenize(word))
							if self.verbose:
								logging.info('pronoun entity found. word: {}, entity: {} - skipping.'.format(word, entity))
						else:
							tok = Token(word, start, end, entity, float(1), entity)
							annotations.append(tok)
							self.anno_counts[entity] += len(word_tokenize(word))
					elif self.verbose:
						logging.warning('invalid annotation found in document {}'.format(idd))

			annotated_doc = Document(idd, annotations)
			annotated_docs.append(annotated_doc)
			if len(annotated_docs) % 100 == 0:
				logging.info('finished {} documents.'.format(len(annotated_docs)))
		reader.close()
		self.annotations = annotated_docs


class Processor:
	def __init__(self, Collection, verbose=False):
		self.text_collection = Collection.documents
		self.annotation_collection = Collection.annotations
		self.verbose = verbose
		self.anno_counts = defaultdict(int)

	def _match_tokens(self, token, anno):
		prefix = None
		if token.start >= anno.start and token.end <= anno.end:
			# check words
			if token.word in anno.word:
				# token match
				if token.score < anno.score:
					# IOB2 encoding
					anno_tokens = word_tokenize(anno.word)
					if len(anno_tokens) >= 1:
						if anno.word.index(token.word) == 0:
							# first word
							prefix = 'B-'
						else:
							prefix = 'I-'
					return prefix
				else:
					# token overlap
					if self.verbose:
						logging.warning('token overlap. Word: {}, Previous: {}:{}, Suggestion: {}:{}'.format(token.word,
																									   token.entity,
																									   token.score,
																									   anno.entity,
																									   anno.score))
					self.entity_overlaps += 1
					self.overlap_tokens_lost[anno.entity] += len(word_tokenize(anno.word))
					return prefix
			else:
				# token not in annotation, index fail
				if self.verbose:
					logging.warning('token not in annotation! Word:{} Annotation:{}'.format(token.word,
																							anno.word))
				self.index_errors += 1
				return prefix
		else:
			return prefix

	def create_conll_format(self, outpath):
		with open(outpath, 'w') as record_file:
			n_docs = 0
			for document in self.text_collection:
				n_docs += 1
				idd = document.id
				annotated_doc = [doc for doc in self.annotation_collection if doc.id == idd][0]
				for token in document.tokens:
					# check annotated docs
					for anno in annotated_doc.tokens:
						# check annotations
						match = self._match_tokens(token, anno)
						if match:
							token.score = anno.score
							token.entity = match + anno.entity
							token.plain_entity = anno.entity
					if (token.start - 1) in (document.sentence_breaks):
						record_file.write('\n')
					# fallback token 0
					if token.word == ' ' or token.word == '':
						token.word = 0
					else:
						self.anno_counts[token.plain_entity] += 1
						line = '{}\t{}\t{}\t{}\t{}\n'.format(token.word,
															 token.start,
															 token.end,
															 token.entity,
															 token.plain_entity)
						record_file.write(line)
				record_file.write('\n')
				if n_docs % 10 == 0:
					logging.info('finished {} documents.'.format(n_docs))

	def annotate_documents(self):
		for n_docs, annotated_document in enumerate(self.annotation_collection):
			doc = [doc for doc in self.text_collection if doc.id == annotated_document.id][0]
			self._match_annotations(annotated_document.tokens, doc.tokens)

	def _match_annotations(self, anno_tokens, doc_tokens):
		for anno in anno_tokens:
			rel_tokens = self._relevant_tokens(anno, doc_tokens)
			if rel_tokens:
				for token in rel_tokens:
					token.score = anno.score
					token.entity = self._IOB2_encoding(anno, token)
					token.plain_entity = anno.entity
			elif self.verbose:
				logging.info('no taggable tokens found, skipping anno: {}'.format(anno.word))

	def _relevant_tokens(self, anno, doc_tokens):
		rel_tokens = []
		for token in doc_tokens:
			if token.start >= anno.start and token.end <= anno.end:
				if token.word in anno.word:
					rel_tokens.append(token)
		if all(token.score < anno.score for token in rel_tokens):
			return rel_tokens
		return None

	def _IOB2_encoding(self, anno, token):
		if anno.word.index(token.word) == 0:
			# first word
			return 'B-' + anno.entity
		else:
			return 'I-' + anno.entity

	def write_conll(self, outpath):
		with open(outpath, 'w') as record_file:
			for n_docs, document in enumerate(self.text_collection):
				for token in document.tokens:
					if (token.start - 1) in (document.sentence_breaks):
						record_file.write('\n')
					if token.word == ' ' or token.word == '':
						continue
					# write token
					self.anno_counts[token.plain_entity] += 1
					line = '{}\t{}\t{}\t{}\t{}\n'.format(token.word,
														 token.start,
														 token.end,
														 token.entity,
														 token.plain_entity)
					record_file.write(line)
				record_file.write('\n')
				if n_docs % 100 == 0:
					logging.info('finished {} documents.'.format(n_docs))
