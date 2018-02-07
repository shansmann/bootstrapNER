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

	def create_conll_format(self, outpath=''):
		for document in self.data_collection.documents:
			# load ann file
			ann_path = str(self.annotation_path + document.id + '.ann')
			annos = self.load_ann_file(ann_path)
			if outpath:
				with open(outpath, 'w') as record_file:
					for token in document.tokens:
						# check annotations
						for anno in annos:
							# check indices
							if token.start >= anno.start and token.end <= anno.end:
								# check words
								if token.word in anno.word:
									if token.entity is not config.OTHER_ENTITY:
										# TODO: how to deal with annotation overlap?
										logging.warning('overriding token: {} with {}'.format(token.entity, anno.entity))
									token.entity = anno.entity
								else:
									logging.warning('annotation and indices do not match! Word:{} Annotation:{}'.format(token.word, anno.word))
						if (token.start - 1) in (document.sentence_breaks):
							record_file.write('\n')
						else:
							line = '{}\t{}\t{}\t{}\n'.format(token.entity, token.start, token.end, token.word)
							record_file.write(line)
			else:
				for token in document.tokens:
					# check annotations
					#TODO: check for annotation overlap!
					for anno in annos:
						if token.start >= anno.start and token.end <= anno.end:
							token.entity = anno.entity
					if (token.start - 1) in (document.sentence_breaks):
						print()
					else:
						print(token.entity, token.start, token.end, token.word)


class HeuristikProcessor(Processor):
	def __init__(self):
		pass

	def create_conll_format(self):
		pass


class AvroProcessor(Processor):
	def create_conll_format(self, outpath=''):
		for document in self.data_collection.documents:
			if outpath:
				with open(outpath, 'w') as record_file:
					for token in document.tokens:
						if (token.start - 1) in (document.sentence_breaks):
							record_file.write('\n')
						else:
							line = '{}\t{}\t{}\t{}\n'.format(token.entity, token.start, token.end, token.word)
							record_file.write(line)
			else:
				for token in document.tokens:
					if (token.start - 1) in (document.sentence_breaks):
						print()
					else:
						print(token.entity, token.start, token.end, token.word)