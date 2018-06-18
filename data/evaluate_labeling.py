from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np
import itertools
from collections import Counter

true = ['O', 'O', 'B-product', 'I-product', 'O', 'O', 'O', 'B-org', 'I-org', 'O']
pred = ['O', 'O', 'B-product', 'B-product', 'O', 'O', 'O', 'B-org', 'B-org', 'O']

#{'O': 0, 'product': 1, 'organization': 2}
labels = ['O', 'product', 'organization']

def read_tagging(path):
	data = []
	with open(path) as f:
		content = [line.strip() for line in f]
		sent = []
		for line in content:
			if line=='':
				data.append(sent)
				sent = []
			else:
				line_stripped = line.split('\t')[-1]
				sent.append(line_stripped)
	return data

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

	plt.title('Histogram {} - token basis'.format(name))
	plt.ylabel('counter')
	plt.bar(indexes, values, width, align='edge')
	plt.xticks(indexes + width * 0.5, labels)

	plt.savefig('histo_{}.pdf'.format(name))
	plt.show()

def compute_precision_token_basis(guessed_sentences, correct_sentences, O_Label):
	assert (len(guessed_sentences) == len(correct_sentences))
	correctCount = 0
	count = 0

	for sentenceIdx in range(len(guessed_sentences)):
		guessed = guessed_sentences[sentenceIdx]
		correct = correct_sentences[sentenceIdx]
		assert (len(guessed) == len(correct))
		for idx in range(len(guessed)):

			if guessed[idx] != O_Label:
				count += 1

				if guessed[idx] == correct[idx]:
					correctCount += 1

	precision = 0
	if count > 0:
		precision = float(correctCount) / count

	return precision

def compute_f1_token_basis(predictions, correct, O_Label):
	prec = compute_precision_token_basis(predictions, correct, O_Label)
	rec = compute_precision_token_basis(correct, predictions, O_Label)

	f1 = 0
	if (rec + prec) > 0:
		f1 = 2.0 * prec * rec / (prec + rec)

	return prec, rec, f1


man_path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/product_corpus_man/full.txt'
auto_path = '/Users/sebastianhansmann/Documents/Code/TU/mt/data/product_corpus_auto/full.txt'

pred = read_tagging(auto_path)
test = read_tagging(man_path)

flat_pred = [item for sublist in pred for item in sublist]
flat_test = [item for sublist in test for item in sublist]

print(Counter(flat_pred))
print(Counter(flat_test))

C = confusion_matrix(flat_test, flat_pred, labels=labels)
plot_confusion_matrix(C, labels, flat_test, flat_pred, normalize=False, title='Confusion matrix: Man- vs. Auto-Labels - token basis')

print(f1_score(flat_test, flat_pred, average='micro', labels=['product', 'organization']))
print(compute_f1_token_basis(pred, test, 'O'))

plot_histo(flat_test, ['product', 'organization'], 'man')
plot_histo(flat_pred, ['product', 'organization'], 'auto')

