import pandas as pd
from keras import backend as K
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.constraints import Constraint
from keras.regularizers import Regularizer
from LinNoise import SGD_lr_mult, ProbabilityConstraint, TraceRegularizer


dataset = pd.read_csv('Iris.csv')

# Splitting the data into training and test test
X = dataset.iloc[:,1:5].values
y = dataset.iloc[:,5].values

encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)
Y = pd.get_dummies(y1).values

model = Sequential()
model.add(Dense(10,input_shape=(4,),activation='tanh', name='layer_1'))
model.add(Dense(8,activation='tanh', name='layer_2'))
model.add(Dense(6,activation='tanh', name='layer_3'))
model.add(Dense(3,activation='softmax', name='layer_4'))
model.add(Dense(3,
                activation='linear',
                kernel_constraint=ProbabilityConstraint(),
                kernel_regularizer=TraceRegularizer(lamb=.01),
                kernel_initializer='identity',
                name='layer_5'))

learning_rate_multipliers = {}
learning_rate_multipliers['layer_1'] = 1
learning_rate_multipliers['layer_2'] = 1
learning_rate_multipliers['layer_3'] = 1
learning_rate_multipliers['layer_4'] = 1
learning_rate_multipliers['layer_5'] = 1

sgd_with_lr_multipliers = SGD_lr_mult(multipliers=learning_rate_multipliers, debug_verbose=True)

model.compile(optimizer=sgd_with_lr_multipliers,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
model.fit(X, Y, epochs=100, verbose=1)

weights_hidden = model.layers[-1].get_weights()[0]
print(weights_hidden)
print(weights_hidden.sum(axis=1))