from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
from collections import Counter

true = ['O', 'O', 'B-product', 'I-product', 'O', 'O', 'O', 'B-org', 'I-org', 'O']
pred = ['O', 'O', 'B-product', 'B-product', 'O', 'O', 'O', 'B-org', 'B-org', 'O']

#{'O': 0, 'product': 1, 'organization': 2}
labels = ['O', 'product', 'organization']

def load_data(test_path, pred_path):
	test = []
	pred = []
	with open(test_path) as f:
		test_content = f.readlines()

	with open(pred_path) as f:
		pred_content = f.readlines()

	for line in test_content:
		label = line.split('\t')[-1].strip()
		test.append(label)

	for line in pred_content:
		label = line.split('\t')[-1].strip()
		pred.append(label)

	return test, pred

def plot_confusion_matrix(cm, classes,
						  man, auto,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('Manual')
	plt.xlabel('Automatic')
	plt.savefig('confusion_matrix_man_auto.pdf')
	plt.show()

def plot_histo(y, labels, name):
	y = Counter(y)
	tmp = {key: value for key, value in y.items() if key in labels}
	labels, values = zip(*Counter(tmp).items())

	indexes = np.arange(len(labels))
	width = 0.5

	plt.title('Histogram {}'.format(name))
	plt.ylabel('counter')
	plt.bar(indexes, values, width, align='edge')
	plt.xticks(indexes + width * 0.5, labels)

	plt.savefig('histo_{}.pdf'.format(name))
	plt.show()


man_path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/product_corpus_man/full.txt'
auto_path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/product_corpus_auto/full.txt'

man, auto = load_data(man_path, auto_path)
print(Counter(man))
print(Counter(auto))
C = confusion_matrix(man, auto, labels=labels)
plot_confusion_matrix(C, labels, man, auto, normalize=False, title='Confusion matrix: Man- vs. Auto-Labels')
plot_histo(man, ['product', 'organization'], 'man')
plot_histo(auto, ['product', 'organization'], 'auto')

