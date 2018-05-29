from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

true = ['O', 'O', 'B-product', 'I-product', 'O', 'O', 'O', 'B-org', 'I-org', 'O']
pred = ['O', 'O', 'B-product', 'B-product', 'O', 'O', 'O', 'B-org', 'B-org', 'O']
labels = ['O', 'B-product', 'I-product', 'B-organization', 'I-organization']

def load_data(test_path, pred_path):
	test = []
	pred = []
	with open(test_path) as f:
		test_content = f.readlines()

	with open(pred_path) as f:
		pred_content = f.readlines()

	for line in test_content:
		test.append(line.split('\t')[0])

	for line in pred_content:
		pred.append(line.split('\t')[0])

	return test, pred

def plot_confusion_matrix(cm, classes,
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
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

man_path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/product_corpus_man/train.txt'
auto_path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/product_corpus_auto/train.txt'

test, pred = load_data(man_path, auto_path)
C = confusion_matrix(test, pred, labels=labels)
plot_confusion_matrix(C, classes=labels, normalize=True, title='Confusion matrix: Man- vs. Auto-Labels')
plt.savefig('confusion_matrix_man_auto.pdf')
plt.show()

