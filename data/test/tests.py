import unittest
import filecmp
import os
import avro.schema
from avro.datafile import DataFileWriter
from avro.io import DatumWriter
import json

from ..datastores import TextCollection, AvroCollection, Processor
from ..config import *

class TestParsingMethods(unittest.TestCase):

	def test_Avro(self):
		json_path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/test/test_doc.json'
		schema_path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/sensor/document.avsc'
		avro_path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/test/test.avro'
		opath = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/test/Avro_test.txt'
		gold = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/test/true_conll.conll'

		self._create_avro(json_path, schema_path, avro_path)

		sensor = AvroCollection('test')
		sensor.parse_text_data(avro_path)
		sensor.parse_annotation_data(avro_path)
		sensor.clean_annotations()

		processor = Processor(sensor, entities=['Material', 'Process', 'Task'])
		processor.create_conll_format(opath)

		self.assertTrue(self._assert_files(gold, opath))


	def test_text(self):
		train = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/test/'
		otrain = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/test/Text_test.txt'
		gold = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/test/true_conll.conll'

		semeval = TextCollection('test')
		semeval.parse_text_data(train)
		semeval.parse_annotation_data(train)
		semeval.clean_annotations()

		processor = Processor(semeval, ['Material', 'Process', 'Task'])
		processor.create_conll_format(otrain)

		self.assertTrue(self._assert_files(gold, otrain))

	def _assert_files(self, gold, path2):
		files_equal = filecmp.cmp(gold, path2, False)
		if files_equal:
			# delete local file
			os.remove(path2)
			return True
		return False

	def _create_avro(self, json_path, schema_path, out_path):
		with open(json_path, 'r') as f:
			data = json.load(f)
		with open(schema_path, 'r') as s:
			schema = avro.schema.Parse(s.read())
		writer = DataFileWriter(open(out_path, "wb"), DatumWriter(), schema)
		writer.append(data)
		writer.close()

if __name__ == '__main__':
	unittest.main()
