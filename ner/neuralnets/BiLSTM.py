"""
A bidirectional LSTM with optional CRF and character-based presentation for NLP sequence tagging.

Author: Nils Reimers
License: Apache-2.0
"""

from __future__ import print_function

import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from keras.utils import *

import os
import sys
import random
import time
import math
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import pylab

#from .keraslayers.ChainCRF import ChainCRF
import util.BIOF1Validation as BIOF1Validation

import sys
if (sys.version_info > (3, 0)):
	import pickle as pkl
else: #Python 2.7 imports
	import cPickle as pkl

class BiLSTM:
	additionalFeatures = []
	learning_rate_updates = {'sgd': {1: 0.1, 3:0.05, 5:0.01} }
	verboseBuild = True

	model = None
	noise_free_model = None
	epoch = 0
	skipOneTokenSentences=True
	num_classes = 0

	dataset = None
	embeddings = None
	labelKey = None
	writeOutput = False
	devAndTestEqual = False
	resultsOut = None
	modelSavePath = None
	maxCharLen = None

	params = {'miniBatchSize': 32, 'dropout': [0.25, 0.25], 'classifier': 'Softmax', 'LSTM-Size': [100], 'optimizer': 'nadam', 'earlyStopping': 5, 'addFeatureDimensions': 10,
				'charEmbeddings': None, 'charEmbeddingsSize':30, 'charFilterSize': 30, 'charFilterLength':3, 'charLSTMSize': 25, 'clipvalue': 0, 'clipnorm': 1 , 'noise': False} #Default params


	def __init__(self, params=None):
		if params != None:
			self.params.update(params)

		logging.info("BiLSTM model initialized with parameters: %s" % str(self.params))

	def setMappings(self, embeddings, mappings):
		self.mappings = mappings
		self.embeddings = embeddings
		self.idx2Word = {v: k for k, v in self.mappings['tokens'].items()}

	def setTrainDataset(self, dataset, labelKey):
		self.dataset = dataset
		self.labelKey = labelKey
		self.label2Idx = self.mappings[labelKey]
		self.idx2Label = {v: k for k, v in self.label2Idx.items()}
		self.mappings['label'] = self.mappings[labelKey]

	def padCharacters(self):
		""" Pads the character representations of the words to the longest word in the dataset """
		#Find the longest word in the dataset
		maxCharLen = 0
		for data in [self.dataset['trainMatrix'], self.dataset['devMatrix'], self.dataset['testMatrix']]:
			for sentence in data:
				for token in sentence['characters']:
					maxCharLen = max(maxCharLen, len(token))

		for data in [self.dataset['trainMatrix'], self.dataset['devMatrix'], self.dataset['testMatrix']]:
			#Pad each other word with zeros
			for sentenceIdx in range(len(data)):
				for tokenIdx in range(len(data[sentenceIdx]['characters'])):
					token = data[sentenceIdx]['characters'][tokenIdx]
					data[sentenceIdx]['characters'][tokenIdx] = np.pad(token, (0,maxCharLen-len(token)), 'constant')

		self.maxCharLen = maxCharLen

	def trainModel(self):
		if self.model == None:
			self.buildModel()

		trainMatrix = self.dataset['trainMatrix']
		self.epoch += 1

		if self.params['optimizer'] in self.learning_rate_updates and self.epoch in self.learning_rate_updates[self.params['optimizer']]:
			K.set_value(self.model.optimizer.lr, self.learning_rate_updates[self.params['optimizer']][self.epoch])
			logging.info("Update Learning Rate to %f" % (K.get_value(self.model.optimizer.lr)))

		iterator = self.online_iterate_dataset(trainMatrix, self.labelKey) if self.params['miniBatchSize'] == 1 else self.batch_iterate_dataset(trainMatrix, self.labelKey)

		for batch in iterator:
			labels = batch[0]
			nnInput = batch[1:]
			self.model.train_on_batch(nnInput, labels)

	def predictLabels(self, sentences, mode=''):
		if self.model == None:
			self.buildModel()

		#noise_free_model = None
		if self.params['noise'] and mode == 'test':
			logging.warning('using model without noise mitigation for test data.')
			self.noise_free_model = self.get_jindal_free_model()

		predLabels = [None]*len(sentences)

		sentenceLengths = self.getSentenceLengths(sentences)

		for senLength, indices in sentenceLengths.items():

			if self.skipOneTokenSentences and senLength == 1:
				if 'O' in self.label2Idx:
					dummyLabel = self.label2Idx['O']
				else:
					dummyLabel = 0
				predictions = [[dummyLabel]] * len(indices) #Tag with dummy label
			else:

				features = ['tokens', 'casing'] + self.additionalFeatures
				inputData = {name: [] for name in features}

				for idx in indices:
					for name in features:
						inputData[name].append(sentences[idx][name])

				for name in features:
					inputData[name] = np.asarray(inputData[name])

				#TODO: add noise-free model initialization
				if self.noise_free_model:
					predictions = self.noise_free_model.predict([inputData[name] for name in features], verbose=False)
				else:
					predictions = self.model.predict([inputData[name] for name in features], verbose=False)

				predictions = predictions.argmax(axis=-1) #Predict classes

			predIdx = 0
			for idx in indices:
				predLabels[idx] = predictions[predIdx]
				predIdx += 1

		self.noise_free_model = None

		return predLabels


	# ------------ Some help functions to train on sentences -----------
	def online_iterate_dataset(self, dataset, labelKey):
		idxRange = list(range(0, len(dataset)))
		random.shuffle(idxRange)

		for idx in idxRange:
				labels = []
				features = ['tokens', 'casing']+self.additionalFeatures

				labels = dataset[idx][labelKey]
				labels = [labels]
				labels = np.expand_dims(labels, -1)

				inputData = {}
				for name in features:
					inputData[name] = np.asarray([dataset[idx][name]])

				yield [labels] + [inputData[name] for name in features]



	def getSentenceLengths(self, sentences):
		sentenceLengths = {}
		for idx in range(len(sentences)):
			sentence = sentences[idx]['tokens']
			if len(sentence) not in sentenceLengths:
				sentenceLengths[len(sentence)] = []
			sentenceLengths[len(sentence)].append(idx)

		return sentenceLengths


	trainSentenceLengths = None
	trainSentenceLengthsKeys = None
	def batch_iterate_dataset(self, dataset, labelKey):
		if self.trainSentenceLengths == None:
			self.trainSentenceLengths = self.getSentenceLengths(dataset)
			self.trainSentenceLengthsKeys = list(self.trainSentenceLengths.keys())

		trainSentenceLengths = self.trainSentenceLengths
		trainSentenceLengthsKeys = self.trainSentenceLengthsKeys

		random.shuffle(trainSentenceLengthsKeys)
		for senLength in trainSentenceLengthsKeys:
			if self.skipOneTokenSentences and senLength == 1: #Skip 1 token sentences
				continue
			sentenceIndices = trainSentenceLengths[senLength]
			random.shuffle(sentenceIndices)
			sentenceCount = len(sentenceIndices)


			bins = int(math.ceil(sentenceCount/float(self.params['miniBatchSize'])))
			binSize = int(math.ceil(sentenceCount / float(bins)))

			numTrainExamples = 0
			for binNr in range(bins):
				tmpIndices = sentenceIndices[binNr*binSize:(binNr+1)*binSize]
				numTrainExamples += len(tmpIndices)


				labels = []
				features = ['tokens', 'casing']+self.additionalFeatures
				inputData = {name: [] for name in features}

				for idx in tmpIndices:
					labels.append(dataset[idx][labelKey])

					for name in features:
						inputData[name].append(dataset[idx][name])

				labels = np.asarray(labels)
				labels = np.expand_dims(labels, -1)

				for name in features:
					inputData[name] = np.asarray(inputData[name])

				yield [labels] + [inputData[name] for name in features]

			assert(numTrainExamples == sentenceCount) #Check that no sentence was missed




	def buildModel(self):
		params = self.params

		if self.params['charEmbeddings'] not in [None, "None", "none", False, "False", "false"]:
			self.padCharacters()

		embeddings = self.embeddings
		casing2Idx = self.dataset['mappings']['casing']

		caseMatrix = np.identity(len(casing2Idx), dtype='float32')

		token_input = Input(shape=(None,),
							name='token_input')

		token_embedding = Embedding(input_dim=embeddings.shape[0],
									output_dim=embeddings.shape[1],
									weights=[embeddings],
									trainable=False,
									name='token_embedding')(token_input)

		casing_input = Input(shape=(None,),
							 name='casing_input')

		casing_embedding = Embedding(input_dim=caseMatrix.shape[0],
									 output_dim=caseMatrix.shape[1],
									 weights=[caseMatrix],
									 trainable=False,
									 name='casing_emd')(casing_input)

		concat_layers = [token_embedding, casing_embedding]
		# :: Character Embeddings ::
		if params['charEmbeddings'] not in [None, "None", "none", False, "False", "false"]:
			charset = self.dataset['mappings']['characters']
			charEmbeddingsSize = params['charEmbeddingsSize']
			maxCharLen = self.maxCharLen
			charEmbeddings= []
			for _ in charset:
				limit = math.sqrt(3.0/charEmbeddingsSize)
				vector = np.random.uniform(-limit, limit, charEmbeddingsSize)
				charEmbeddings.append(vector)

			charEmbeddings[0] = np.zeros(charEmbeddingsSize) #Zero padding
			charEmbeddings = np.asarray(charEmbeddings)

			character_input = Input(shape=(None, maxCharLen),
									name='character_input')
			character_embedding = TimeDistributed(Embedding(input_dim=charEmbeddings.shape[0],
															output_dim=charEmbeddings.shape[1],
															weights=[charEmbeddings],
															trainable=True,
															mask_zero=True),
												  input_shape=(None, maxCharLen),
												  name='char_emd')(character_input)

			charLSTMSize = params['charLSTMSize']
			character_lstm = TimeDistributed(Bidirectional(LSTM(charLSTMSize,
																return_sequences=False)),
											 name="char_lstm")(character_embedding)

			if self.additionalFeatures == None:
				self.additionalFeatures = []

			self.additionalFeatures.append('characters')
			concat_layers.append(character_lstm)

		merged = keras.layers.concatenate(concat_layers,
										  name='concat_layer')

		bi_lstm_1 = Bidirectional(LSTM(params['LSTM-Size'][0],
									   return_sequences=True,
									   dropout_W=params['dropout'][0],
									   dropout_U=params['dropout'][1]),
								  name="BiLSTM_1")(merged)

		bi_lstm_2 = Bidirectional(LSTM(params['LSTM-Size'][1],
									   return_sequences=True,
									   dropout_W=params['dropout'][0],
									   dropout_U=params['dropout'][1]),
								  name="BiLSTM_2")(bi_lstm_1)

		self.num_classes = len(self.dataset['mappings'][self.labelKey])

		output = TimeDistributed(Dense(self.num_classes,
									   activation='softmax'),
								 name='softmax_output')(bi_lstm_2)

		# jindals noise model
		if self.params['noise']:
			hadamard_jindal = TimeDistributed(Dropout(.1),
											  name='hadamard_jindal')(output)
			output = TimeDistributed(Dense(self.num_classes,
											activation='softmax',
											bias=False,
											weights=[np.identity(10, dtype='float32')]),
									  name='softmax_jindal')(hadamard_jindal)

		model = Model(inputs=[token_input, casing_input, character_input], outputs=output)
		#model = Model(inputs=[token_embedding, casing_embedding], outputs=output)

		optimizerParams = {}
		if 'clipnorm' in self.params and self.params['clipnorm'] != None and  self.params['clipnorm'] > 0:
			optimizerParams['clipnorm'] = self.params['clipnorm']

		if 'clipvalue' in self.params and self.params['clipvalue'] != None and  self.params['clipvalue'] > 0:
			optimizerParams['clipvalue'] = self.params['clipvalue']

		if params['optimizer'].lower() == 'adam':
			opt = Adam(**optimizerParams)
		elif params['optimizer'].lower() == 'nadam':
			opt = Nadam(**optimizerParams)
		elif params['optimizer'].lower() == 'rmsprop':
			opt = RMSprop(**optimizerParams)
		elif params['optimizer'].lower() == 'adadelta':
			opt = Adadelta(**optimizerParams)
		elif params['optimizer'].lower() == 'adagrad':
			opt = Adagrad(**optimizerParams)
		elif params['optimizer'].lower() == 'sgd':
			opt = SGD(lr=0.1, **optimizerParams)

		model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)

		self.model = model

		#new_model = Model(inputs=model.inputs, outputs=model.layers[-3].output)
		#new_model.summary()
		#plot_model(new_model, to_file='new_model.png', show_shapes=True, show_layer_names=True)

		if self.verboseBuild:
			model.summary()
			logging.debug(model.get_config())
			logging.debug("Optimizer: %s, %s" % (str(type(opt)), str(opt.get_config())))
			#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

	def storeResults(self, resultsFilepath):
		if resultsFilepath != None:
			directory = os.path.dirname(resultsFilepath)
			if not os.path.exists(directory):
				os.makedirs(directory)

			self.resultsOut = open(resultsFilepath, 'w')
		else:
			self.resultsOut = None

	def get_jindal_free_model(self):
		new_model = Model(inputs=self.model.inputs, outputs=self.model.layers[-3].output)
		return new_model

	def evaluate(self, epochs):
		logging.info("%d train sentences" % len(self.dataset['trainMatrix']))
		logging.info("%d dev sentences" % len(self.dataset['devMatrix']))
		logging.info("%d test sentences" % len(self.dataset['testMatrix']))

		devMatrix = self.dataset['devMatrix']
		testMatrix = self.dataset['testMatrix']

		total_train_time = 0
		max_dev_score = 0
		max_test_score = 0
		no_improvement_since = 0

		for epoch in range(epochs):
			sys.stdout.flush()
			logging.info("--------- Epoch %d -----------" % (epoch+1))

			start_time = time.time()
			self.trainModel()
			time_diff = time.time() - start_time
			total_train_time += time_diff
			logging.info("%.2f sec for training (%.2f total)" % (time_diff, total_train_time))


			start_time = time.time()
			dev_score, test_score = self.computeScores(devMatrix, testMatrix)

			if dev_score > max_dev_score:
				no_improvement_since = 0
				max_dev_score = dev_score
				max_test_score = test_score

				if self.modelSavePath != None:
					savePath = self.modelSavePath.replace("[DevScore]", "%.4f" % dev_score).replace("[TestScore]", "%.4f" % test_score).replace("[Epoch]", str(epoch))

					directory = os.path.dirname(savePath)
					if not os.path.exists(directory):
						os.makedirs(directory)

					if not os.path.isfile(savePath):
						self.model.save(savePath, False)


						#self.save_dict_to_hdf5(self.mappings, savePath, 'mappings')

						import json
						import h5py
						mappingsJson = json.dumps(self.mappings)
						with h5py.File(savePath, 'a') as h5file:
							h5file.attrs['mappings'] = mappingsJson
							h5file.attrs['additionalFeatures'] = json.dumps(self.additionalFeatures)
							h5file.attrs['maxCharLen'] = str(self.maxCharLen)

						#mappingsOut = open(savePath+'.mappings', 'wb')
						#pkl.dump(self.dataset['mappings'], mappingsOut)
						#mappingsOut.close()
					else:
						logging.info("Model", savePath, "already exists")
			else:
				no_improvement_since += 1


			if self.resultsOut != None:
				self.resultsOut.write("\t".join(map(str, [epoch+1, dev_score, test_score, max_dev_score, max_test_score])))
				self.resultsOut.write("\n")
				self.resultsOut.flush()

			logging.info("Max: %.4f on dev; %.4f on test" % (max_dev_score, max_test_score))
			logging.info("%.2f sec for evaluation" % (time.time() - start_time))

			if self.params['earlyStopping'] > 0 and no_improvement_since >= self.params['earlyStopping']:
				logging.info("!!! Early stopping, no improvement after "+str(no_improvement_since)+" epochs !!!")
				break
		# unplug model
		#logging.warning(self.model.summary())
		#self.remove_jindal()
		#logging.warning(self.model.summary())
		# evaluate model on test data
		#dev_score, test_score = self.computeScores(devMatrix, testMatrix)
		#logging.info("%.4f on test" % (test_score))
		if self.verboseBuild and self.params["noise"]:
			self.plot_noise_dists(max_test_score)

	def plot_noise_dists(self, test_score):
		weights = self.model.layers[-1].get_weights()[0]
		pylab.imshow(weights, cmap='hot', interpolation='nearest')
		pylab.xticks(np.arange(0, self.num_classes))
		pylab.yticks(np.arange(0, self.num_classes))
		pylab.colorbar()
		pylab.title('learned noise - jindal - f1: {}'.format(test_score))

		#plt.tight_layout()
		pylab.savefig('noise_dist_learned_f1_{}.pdf'.format(test_score))
		#plt.close(fig)

	def computeScores(self, devMatrix, testMatrix):
		if self.labelKey.endswith('_BIO') or self.labelKey.endswith('_IOB') or self.labelKey.endswith('_IOBES'):
			logging.info("computing F1 Scores.")
			return self.computeF1Scores(devMatrix, testMatrix)
		else:
			return self.computeAccScores(devMatrix, testMatrix)

	def computeF1Scores(self, devMatrix, testMatrix):
		dev_pre, dev_rec, dev_f1 = self.computeF1(devMatrix, 'dev')
		logging.info("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (dev_pre, dev_rec, dev_f1))

		if self.devAndTestEqual:
			test_pre, test_rec, test_f1 = dev_pre, dev_rec, dev_f1
		else:
			test_pre, test_rec, test_f1 = self.computeF1(testMatrix, 'test')
		logging.info("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (test_pre, test_rec, test_f1))

		return dev_f1, test_f1

	def computeAccScores(self, devMatrix, testMatrix):
		dev_acc = self.computeAcc(devMatrix)
		test_acc = self.computeAcc(testMatrix)

		logging.info("Dev-Data: Accuracy: %.4f" % (dev_acc))
		logging.info("Test-Data: Accuracy: %.4f" % (test_acc))

		return dev_acc, test_acc


	def tagSentences(self, sentences):

		#Pad characters
		if 'characters' in self.additionalFeatures:
			maxCharLen = self.maxCharLen
			for sentenceIdx in range(len(sentences)):
				for tokenIdx in range(len(sentences[sentenceIdx]['characters'])):
					token = sentences[sentenceIdx]['characters'][tokenIdx]
					sentences[sentenceIdx]['characters'][tokenIdx] = np.pad(token, (0, maxCharLen-len(token)), 'constant')


		paddedPredLabels = self.predictLabels(sentences)
		predLabels = []
		for idx in range(len(sentences)):
			unpaddedPredLabels = []
			for tokenIdx in range(len(sentences[idx]['tokens'])):
				if sentences[idx]['tokens'][tokenIdx] != 0: #Skip padding tokens
					unpaddedPredLabels.append(paddedPredLabels[idx][tokenIdx])

			predLabels.append(unpaddedPredLabels)


		idx2Label = {v: k for k, v in self.mappings['label'].items()}
		labels = [[idx2Label[tag] for tag in tagSentence] for tagSentence in predLabels]

		return labels

	def computeF1(self, sentences, name=''):
		correctLabels = []
		predLabels = []
		paddedPredLabels = self.predictLabels(sentences, name)

		for idx in range(len(sentences)):
			unpaddedCorrectLabels = []
			unpaddedPredLabels = []
			for tokenIdx in range(len(sentences[idx]['tokens'])):
				if sentences[idx]['tokens'][tokenIdx] != 0: #Skip padding tokens
					unpaddedCorrectLabels.append(sentences[idx][self.labelKey][tokenIdx])
					unpaddedPredLabels.append(paddedPredLabels[idx][tokenIdx])

			correctLabels.append(unpaddedCorrectLabels)
			predLabels.append(unpaddedPredLabels)


		encodingScheme = self.labelKey[self.labelKey.index('_')+1:]

		pre, rec, f1 = BIOF1Validation.compute_f1(predLabels, correctLabels, self.idx2Label, 'O', encodingScheme)
		pre_b, rec_b, f1_b = BIOF1Validation.compute_f1(predLabels, correctLabels, self.idx2Label, 'B', encodingScheme)


		if f1_b > f1:
			logging.debug("Setting incorrect tags to B yields improvement from %.4f to %.4f" % (f1, f1_b))
			pre, rec, f1 = pre_b, rec_b, f1_b


		if self.writeOutput:
			self.writeOutputToFile(sentences, predLabels, '%.4f_%s' % (f1, name))
		return pre, rec, f1

	def writeOutputToFile(self, sentences, predLabels, name):
			outputName = 'tmp/'+name
			fOut = open(outputName, 'w')

			for sentenceIdx in range(len(sentences)):
				for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
					token = self.idx2Word[sentences[sentenceIdx]['tokens'][tokenIdx]]
					label = self.idx2Label[sentences[sentenceIdx][self.labelKey][tokenIdx]]
					predLabel = self.idx2Label[predLabels[sentenceIdx][tokenIdx]]

					fOut.write("\t".join([token, label, predLabel]))
					fOut.write("\n")

				fOut.write("\n")

			fOut.close()



	def computeAcc(self, sentences):
		correctLabels = [sentences[idx][self.labelKey] for idx in range(len(sentences))]
		predLabels = self.predictLabels(sentences)

		numLabels = 0
		numCorrLabels = 0
		for sentenceId in range(len(correctLabels)):
			for tokenId in range(len(correctLabels[sentenceId])):
				numLabels += 1
				if correctLabels[sentenceId][tokenId] == predLabels[sentenceId][tokenId]:
					numCorrLabels += 1


		return numCorrLabels/float(numLabels)

	def loadModel(self, modelPath="/mnt/hdd/experiments/shansmann/bootstrapNER/ner/models/sensor_corpus_auto/NER_BIO/0.0634_0.0345_7.h5"):
		import h5py
		import json
		from neuralnets.keraslayers.ChainCRF import create_custom_objects

		model = keras.models.load_model(modelPath, custom_objects=create_custom_objects())

		with h5py.File(modelPath, 'r') as f:
			mappings = json.loads(f.attrs['mappings'])
			if 'additionalFeatures' in f.attrs:
				self.additionalFeatures = json.loads(f.attrs['additionalFeatures'])

			if 'maxCharLen' in f.attrs and f.attrs['maxCharLen'] != None and f.attrs['maxCharLen'] != 'None':
				self.maxCharLen = int(f.attrs['maxCharLen'])

		self.model = model
		self.setMappings(None, mappings)
		self.plot_noise_dists('0.0345')

