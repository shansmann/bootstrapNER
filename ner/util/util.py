from sklearn.metrics import f1_score
from keras import backend as K

def f1_train(y_true, y_pred):
    return f1_score(K.eval(y_true), K.eval(y_pred), average='micro')

