from __future__ import print_function
import os
import logging
import sys
import neuralnets.BiLSTM
import util.preprocessing


# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

######################################################
#
# Data preprocessing
#
######################################################

# :: Train / Dev / Test-Files ::
datasetName = 'sensor_corpus'
dataColumns = {0:'tokens', 4:'NER'} #Tab separated columns, column 1 contains the token, 2 the NER using BIO-encoding
labelKey = 'NER'

embeddingsPath = 'glove.840B.300d.txt' #glove word embeddings

#Parameters of the network
params = {'dropout': [0.25, 0.25],
          'classifier': 'softmax',
          'LSTM-Size': [100,75],
          'optimizer': 'nadam',
          'charEmbeddings': 'LSTM',
          'miniBatchSize': 128,
          'noise': True}

frequencyThresholdUnknownTokens = 5 #If a token that is not in the pre-trained embeddings file appears at least 50 times in the train.txt, then a new embedding is generated for this word
training_embeddings_only = False

datasetFiles = [
        (datasetName, dataColumns),
    ]

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = util.preprocessing.perpareDataset(embeddingsPath, datasetFiles, frequencyThresholdUnknownTokens, training_embeddings_only)

######################################################
#
# The training of the network starts here
#
######################################################

#Load the embeddings and the dataset
embeddings, word2Idx, datasets = util.preprocessing.loadDatasetPickle(pickleFile)
data = datasets[datasetName]

logging.info("Dataset: {}".format(datasetName))
logging.info(data['mappings'].keys())
logging.info(data['mappings'][labelKey])

model = neuralnets.BiLSTM.BiLSTM(params)
model.setMappings(embeddings, data['mappings'])
model.setTrainDataset(data, labelKey)
model.verboseBuild = True
model.buildModel()
model.modelSavePath = "models/%s/%s/[DevScore]_[Epoch].h5" % (datasetName, labelKey) #Enable this line to save the model to the disk
model.evaluate(1)
