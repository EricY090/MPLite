import os
import _pickle as pickle

from metrics import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import numpy as np

mimic3_path = os.path.join('data', 'mimic3')
encoded_path = os.path.join(mimic3_path, 'encoded')
standard_path = os.path.join(mimic3_path, 'standard')


def load_data():
    labs_dataset = pickle.load(open(os.path.join(standard_path, 'labs_dataset.pkl'), 'rb'))
    return labs_dataset


if __name__ == '__main__':
    labs_dataset = load_data()
    train_labs_data, valid_labs_data, test_labs_data = labs_dataset['train_labs_data'], labs_dataset['valid_labs_data'], labs_dataset['test_labs_data']
    

    (train_single_lab_x, train_single_lab_y) = train_labs_data
    (valid_single_lab_x, valid_single_lab_y) = valid_labs_data
    (test_single_lab_x, test_single_lab_y) = test_labs_data
    
    train_single_lab_x = tf.convert_to_tensor(train_single_lab_x, dtype=tf.float32)
    train_single_lab_y = tf.convert_to_tensor(train_single_lab_y, dtype=tf.float32)
    valid_single_lab_x = tf.convert_to_tensor(valid_single_lab_x, dtype=tf.float32)
    valid_single_lab_y = tf.convert_to_tensor(valid_single_lab_y, dtype=tf.float32)
    test_single_lab_x = tf.convert_to_tensor(test_single_lab_x, dtype=tf.float32)
    test_single_lab_y = tf.convert_to_tensor(test_single_lab_y, dtype=tf.float32)
    
    print(len(train_single_lab_x), len(valid_single_lab_x), len(test_single_lab_x))     ### 39082, 2300, 4597
    
    
    item_num = len(train_single_lab_x[0])       #697
    code_num = len(train_single_lab_y[0])       
    
    model = keras.Sequential()
    model.add(keras.Input(shape = (item_num, )))
    model.add(layers.Dense(200, activation = "relu"))
    model.add(layers.Dense(code_num, activation = "sigmoid"))
    
    model.compile(optimizer='adam',
              loss= tf.keras.losses.BinaryCrossentropy(from_logits=False), 
              metrics=[tf.keras.metrics.Recall(top_k = 20, thresholds = 0.1),
                       tf.keras.metrics.Recall(top_k = 40, thresholds = 0.1),
                       tfa.metrics.F1Score(num_classes = code_num, average="weighted", threshold = 0.1, name = "weighted_f1")])
    
    model.fit(train_single_lab_x, train_single_lab_y, validation_data = (valid_single_lab_x, valid_single_lab_y), batch_size=32, epochs = 100, verbose = 2)
    model.evaluate(test_single_lab_x, test_single_lab_y)
    
    ### Save Pre-trained Model
    pre_trained = os.path.join(mimic3_path, 'Pre-trained')
    if not os.path.exists(pre_trained):
        os.makedirs(pre_trained)
    model.save(pre_trained)
    np.save(os.path.join(pre_trained, 'saved_weights.npy'), model.layers[0].get_weights()[0])
    np.save(os.path.join(pre_trained, 'saved_bias.npy'), model.layers[0].get_weights()[1])

    
    