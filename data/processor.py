import xmltodict
from nltk.tokenize import word_tokenize

from objects import Word, Annotation, Document, DataCollection
import logging, coloredlogs
import config

coloredlogs.install()


class Processor:
	def __init__(self, DataCollection):
		self.data_collection = DataCollection

	def create_conll_format(self):
		for document in self.data_collection.documents:
			for token in document.tokens:
				if (token.start-1) in (document.sentence_breaks):
					print()
				else:
					print(token.entity, token.start, token.end, token.word)


class AnnProcessor(Processor):
	def __init__(self, DataCollection, annotation_path):
		self.data_collection = DataCollection
		self.annotation_path = annotation_path
		self.anno_ending = '.ann'

	def load_ann_file(self, path):
		annos = []
		f_anno = open(path, 'rU')
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
					annos.append(Annotation(surface, int(start), int(end), keytype))
		f_anno.close()
		return annos

	def return_unique_annos(self, anno_type, unique_n=300):
		unique_annos = set()
		for document in self.data_collection.documents:
			# load ann file
			ann_path = str(self.annotation_path + document.id + self.anno_ending)
			annos = self.load_ann_file(ann_path)
			for anno in annos:
				if anno.entity == anno_type:
					if anno.word in unique_annos:
						continue
					else:
						unique_annos.add(anno.word)
						if len(unique_annos) >= unique_n:
							logging.info('{} unique entities found.'.format(unique_n))
							return list(unique_annos)
		return list(unique_annos)

	def create_conll_format(self, outpath):
		ent_overlaps = 0
		with open(outpath, 'w') as record_file:
			for document in self.data_collection.documents:
				# load ann file
				ann_path = str(self.annotation_path + document.id + self.anno_ending)
				annos = self.load_ann_file(ann_path)
				for token in document.tokens:
					# check annotations
					for anno in annos:
						# check indices
						if token.start >= anno.start and token.end <= anno.end:
							# check words
							if token.word in anno.word:
								# token match
								if token.entity is config.OTHER_ENTITY:
									# IOB encoding
									anno_tokens = word_tokenize(anno.word)
									if len(anno_tokens) > 1:
										if anno.word.index(token.word) == 0:
											prefix = 'B-'
										else:
											prefix = 'I-'
									else:
										prefix = 'B-'
									# label token
									token.entity = prefix + anno.entity
								else:
									# TODO: Token overlap for now ignored
									ent_overlaps += 1
									continue
									logging.warning('problem in document: {}'.format(document.id + self.anno_ending))
									logging.warning('overriding token: {} with {}'.format(token.entity, anno.entity))
							else:
								# token mismatch
								logging.warning('problem in document: {}'.format(document.id + self.anno_ending))
								logging.warning(
									'annotation and indices do not match! Word:{} Annotation:{}'.format(token.word,
																										anno.word))
					if (token.start - 1) in (document.sentence_breaks):
						record_file.write('\n')
					line = '{}\t{}\t{}\t{}\n'.format(token.entity, token.start, token.end, token.word)
					record_file.write(line)
		logging.info('{} entity overlaps occured.'.format(ent_overlaps))

class XMLProcessor(AnnProcessor):
	def __init__(self, DataCollection, annotation_path):
		self.data_collection = DataCollection
		self.annotation_path = annotation_path
		self.anno_ending = '.xml'

	def load_ann_file(self, path):
		annos = []
		with open(path) as fd:
			doc = xmltodict.parse(fd.read())
			try:
				entities = doc['Document']['Entities']['Entity']
			except:
				logging.warning('no entities found in: {}'.format(path))
				return annos
			for elem in entities:
				try:
					word = elem['@surface']
					start = elem['@startPos']
					end = elem['@endPos']
					typ = elem['@type']
					if word and not typ=='trigger':
						annos.append(Annotation(word, int(start), int(end), typ))
				except:
					continue
		return annos
	"""
	def create_conll_format(self, outpath):
		with open(outpath, 'w') as record_file:
			for document in self.data_collection.documents:
				# load ann file
				ann_path = str(self.annotation_path + document.id + self.anno_ending)
				annos = self.load_ann_file(ann_path)
				for token in document.tokens:
					# check annotations
					for anno in annos:
						# check indices
						if token.start >= anno.start and token.end <= anno.end:
							# check words
							if token.word in anno.word:
								if token.entity is not config.OTHER_ENTITY:
									# TODO: how to deal with annotation overlap?
									logging.warning('problem in document: {}'.format(document.id + self.anno_ending))
									logging.warning('overriding token: {} with {}'.format(token.entity, anno.entity))
								token.entity = anno.entity
							else:
								logging.warning('problem in document: {}'.format(document.id + self.anno_ending))
								logging.warning('annotation and indices do not match! Word:{} Annotation:{}'.format(token.word, anno.word))
					if (token.start - 1) in (document.sentence_breaks):
						record_file.write('\n')
					line = '{}\t{}\t{}\t{}\n'.format(token.entity, token.start, token.end, token.word)
					record_file.write(line)
	"""


class HeuristikProcessor(Processor):
	def __init__(self):
		pass

	def create_conll_format(self):
		pass


class AvroProcessor(Processor):
	def create_conll_format(self, outpath):
		with open(outpath, 'w') as record_file:
			for document in self.data_collection.documents:
					for token in document.tokens:
						if (token.start - 1) in (document.sentence_breaks):
							record_file.write('\n')
						line = '{}\t{}\t{}\t{}\n'.format(token.entity, token.start, token.end, token.word)
						record_file.write(line)
