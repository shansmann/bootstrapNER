"""
this script implements jindal's dropout regularization for learning deep networks with
noisy labels

author: mail@sebastianhansmann.com
"""
from numpy.random import seed
from tensorflow import set_random_seed
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# config
batch_size = 128
num_classes = 10
epochs = 20
noise = .05
drop = .1
seed(7)
set_random_seed(7)


def get_noise_dist(p):
    phi = (1 - p) * np.identity(10) + (p / 10) * np.ones((10, 1)) * np.ones((1, 10))
    return phi

def apply_noise(data, noise):
    cnt = 0
    new_data = []
    phi = get_noise_dist(noise)
    for sample in data:
        val = np.random.choice(np.arange(0, 10), p=phi[sample])
        new_data.append(val)
        if sample != val:
            cnt +=1
    print('flipped {}% of labels.'.format(cnt/len(data)))
    return new_data

def build_model(noise=False):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,), name='hidden1'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', name='hidden2'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax', name='output'))
    if noise:
        # noise model dropout
        model.add(Dense(num_classes, name='linear_jindal', bias=False, weights=[np.identity(10, dtype='float32')]))
        model.add(Dropout(drop, name='hadamard_jindal'))
        model.add(Dense(num_classes, name='softmax_jindal', activation='softmax'))
    return model

def train_model(model, x_train, y_train, x_dev, y_dev):
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        nb_epoch=epochs,
                        verbose=1,
                        validation_data=(x_dev, y_dev))
    return history

def plot_noise_dists(noise, model, drop, epochs):
    phi = get_noise_dist(noise)
    weights_hidden = model.layers[-3].get_weights()[0]

    plt.subplot(2, 1, 1)
    plt.imshow(phi, cmap='hot', interpolation='nearest', vmax=1, vmin=0)
    plt.xticks(np.arange(0, 10))
    plt.yticks(np.arange(0, 10))
    plt.colorbar()
    plt.title('true noise - jindal')

    plt.subplot(2, 1, 2)
    plt.imshow(weights_hidden, cmap='hot', interpolation='nearest')
    plt.xticks(np.arange(0, 10))
    plt.yticks(np.arange(0, 10))
    plt.colorbar()
    plt.title('linear layer - jindal')

    plt.tight_layout()
    plt.savefig('noise_dist_n{}_d{}_e{}_var2.pdf'.format(noise, drop, epochs))
    plt.show()

def prepare_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # splitting test data into dev and test set
    x_test, x_dev, y_test, y_dev = train_test_split(x_test, y_test, test_size=0.2, random_state=7)

    print(x_train.shape[0], 'train samples')
    print(x_dev.shape[0], 'dev samples')
    print(x_test.shape[0], 'test samples')

    # flip training labels
    y_train = apply_noise(y_train, noise)

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_dev = np_utils.to_categorical(y_dev, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_dev, y_dev, x_test, y_test


x_train, y_train, x_dev, y_dev, x_test, y_test = prepare_data()

model = build_model(noise=True)

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = train_model(model, x_train, y_train, x_dev, y_dev)

plot_noise_dists(noise, model, drop, epochs)

# retrain model without noise extension
for layer in [0, 2, 4]:
    weightsAndBiases = model.layers[layer].get_weights()
    if len(weightsAndBiases) == 2:
        print('saving weights & bias from layer', model.layers[layer].name)
        np.save('layer{}_weights.npy'.format(layer), weightsAndBiases[0])  # .npy extension is added if not given
        np.save('layer{}_bias.npy'.format(layer), weightsAndBiases[1])
    else:
        print(weightsAndBiases)
        print(model.layers[layer].name)

model_wo_noise = build_model(noise=False)

weights0 = np.load('layer0_weights.npy')
biases0 = np.load('layer0_bias.npy')
weights2 = np.load('layer2_weights.npy')
biases2 = np.load('layer2_bias.npy')
weights4 = np.load('layer4_weights.npy')
biases4= np.load('layer4_bias.npy')

model_wo_noise.layers[0].set_weights([weights0,biases0])
model_wo_noise.layers[0].trainable = False
model_wo_noise.layers[2].set_weights([weights2,biases2])
model_wo_noise.layers[2].trainable = False
model_wo_noise.layers[4].set_weights([weights4,biases4])
model_wo_noise.layers[4].trainable = False

model_wo_noise.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

score = model_wo_noise.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])