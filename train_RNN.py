import os
import _pickle as pickle

from metrics import *
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import numpy as np
from utils import DataGenerator
from metrics import EvaluateCodesCallBackRNN
from loss import medical_codes_loss


mimic3_path = os.path.join('data', 'mimic3')
standard_path = os.path.join(mimic3_path, 'standard')
pre_trained = os.path.join(mimic3_path, 'Pre-trained')

def load_data():
    codes_dataset = pickle.load(open(os.path.join(standard_path, 'codes_dataset.pkl'), 'rb'))
    return codes_dataset


def convert_hot(x):
    result = np.zeros_like(x ,dtype=int)
    for i, examples in enumerate(x):
        for j, step in enumerate(examples):
            if np.all(x[i][j]==0): break
            hot_index = [t-1 for t in x[i][j] if t!= 0]
            result[i][j][hot_index] = 1
    return result


class RNN(keras.layers.Layer):
    def __init__(self, unit, steps, input_dim):
        super().__init__()
        self.block0 = tf.keras.layers.Masking(mask_value=0, input_shape = (steps, input_dim))
        self.block1 = tf.keras.layers.GRU(unit)

    def call(self, codes):
        x = self.block0(codes)
        x = self.block1(x)
        return x
        
        
class Pre_trained(keras.layers.Layer):
    def __init__(self, units, w, b):
        super().__init__()
        self.dense = keras.layers.Dense(units, kernel_initializer= tf.constant_initializer(w), bias_initializer= tf.constant_initializer(b), activation=None, trainable = False)

    def call(self, x, lab):
        lab = self.dense(lab)
        output = tf.concat([x, lab], 1)
        return output

class Classifier(keras.layers.Layer):
    def __init__(self, output_dim, name='classifier'):
        super().__init__(name=name)
        self.dense = keras.layers.Dense(output_dim, activation=None)
        self.dropout = tf.keras.layers.Dropout(0.4)

    def call(self, x):
        x = self.dropout(x)
        output = self.dense(x)
        return output


class Model(keras.Model):
    def __init__(self, output_dim, w, b, steps, input_dim, use_lab):
        super().__init__()
        self.use_lab = use_lab
        self.rnn = RNN(128, steps, input_dim)
        if self.use_lab:
            self.trained = Pre_trained(200, w, b)
        self.classifier = Classifier(output_dim)
    
    def call(self, inputs):
        x = self.rnn(inputs['codes_x'])
        if self.use_lab:
            x = self.trained(x, inputs['lab_x'])
        output = self.classifier(x)
        return output

def lr_schedule_fn(epoch, lr):
        if epoch < 40:
            lr = 0.001
        elif epoch < 100:
            lr = 0.0001
        elif epoch < 200:
            lr = 0.00001
        else:
            lr = 0.000001
        return lr
    

if __name__ == '__main__':
    use_lab = True
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    codes_dataset = load_data()
    train_codes_data, valid_codes_data, test_codes_data = codes_dataset['train_codes_data'], codes_dataset['valid_codes_data'], codes_dataset['test_codes_data']
    

    (train_codes_x, train_codes_y, train_lab_x, train_visit_lens) = train_codes_data
    (valid_codes_x, valid_codes_y, valid_lab_x, valid_visit_lens) = valid_codes_data
    (test_codes_x, test_codes_y, test_lab_x, test_visit_lens) = test_codes_data
    train_codes_x, valid_codes_x, test_codes_x = convert_hot(train_codes_x), convert_hot(valid_codes_x), convert_hot(test_codes_x)
    print(train_codes_x.shape, valid_codes_x.shape, test_codes_x.shape)
    
    train_codes_x = tf.convert_to_tensor(train_codes_x, dtype=tf.float32)
    train_codes_y = tf.convert_to_tensor(train_codes_y, dtype=tf.float32)
    valid_codes_x = tf.convert_to_tensor(valid_codes_x, dtype=tf.float32)
    valid_codes_y = tf.convert_to_tensor(valid_codes_y, dtype=tf.float32)
    test_codes_x = tf.convert_to_tensor(test_codes_x, dtype=tf.float32)
    test_codes_y = tf.convert_to_tensor(test_codes_y, dtype=tf.float32)
    train_lab_x = tf.convert_to_tensor(train_lab_x, dtype=tf.float32)
    valid_lab_x = tf.convert_to_tensor(valid_lab_x, dtype=tf.float32)
    test_lab_x = tf.convert_to_tensor(test_lab_x, dtype=tf.float32)
    

    
    test_codes_gen = DataGenerator([test_codes_x, test_lab_x], shuffle=False)
    test_callback = EvaluateCodesCallBackRNN(test_codes_gen, test_codes_y)
    
    ### RNN: diagnoses with labs -> diagnoses
    max_admission = len(train_codes_x[0])
    in_dim = len(train_codes_x[0][0])
    out_dim = len(train_codes_y[0])
    pre_trained_model = keras.models.load_model(pre_trained)
    weights = pre_trained_model.layers[0].get_weights()
    weight, bias = weights[0], weights[1]
    lr_scheduler = LearningRateScheduler(lr_schedule_fn)
    
    model = Model(out_dim, weight, bias, max_admission, in_dim, use_lab)
    model.compile(optimizer='adam', loss= medical_codes_loss)
    
    #validation_data = ({'codes_x': valid_codes_x, 'lab_x': valid_lab_x}, valid_codes_y), 
    model.fit(x = {'codes_x': train_codes_x, 'lab_x': train_lab_x}, y = train_codes_y, 
                    batch_size=64, epochs = 100, callbacks=[test_callback, lr_scheduler], verbose = 2)
    #model.evaluate({'codes_x': test_codes_x, 'lab_x': test_lab_x}, test_codes_y)
    model.summary()
    
    