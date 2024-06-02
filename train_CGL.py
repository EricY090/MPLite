### Original CGL without lab. Multiple examples per patients
import os
import random
import _pickle as pickle

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import tensorflow_addons as tfa
import numpy as np

from models.model import CGL
from loss import medical_codes_loss
from metrics import EvaluateCodesCallBack
from utils import DataGenerator
from sklearn.utils import shuffle


seed = 6669
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


mimic3_path = os.path.join('data', 'mimic3')
encoded_path = os.path.join(mimic3_path, 'encoded')
standard_path = os.path.join(mimic3_path, 'standard')
pre_trained = os.path.join(mimic3_path, 'Pre-trained')


def load_data() -> (tuple, tuple, dict):
    code_map = pickle.load(open(os.path.join(encoded_path, 'code_map.pkl'), 'rb'))
    codes_dataset = pickle.load(open(os.path.join(standard_path, 'codes_dataset.pkl'), 'rb'))
    auxiliary = pickle.load(open(os.path.join(standard_path, 'auxiliary.pkl'), 'rb'))
    return code_map, codes_dataset, auxiliary


def historical_hot(code_x, code_num):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, x in enumerate(code_x):
        for code in x:
            result[i][code - 1] = 1
    return result


if __name__ == '__main__':
    use_lab = False
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    code_map, codes_dataset, auxiliary = load_data()
    train_codes_data, valid_codes_data, test_codes_data = codes_dataset['train_codes_data'], codes_dataset['valid_codes_data'], codes_dataset['test_codes_data']

    (train_codes_x, train_codes_y, train_lab_x, train_visit_lens) = train_codes_data
    (valid_codes_x, valid_codes_y, valid_lab_x, valid_visit_lens) = valid_codes_data
    (test_codes_x, test_codes_y, test_lab_x, test_visit_lens) = test_codes_data
    code_levels, patient_code_adj, code_code_adj = auxiliary['code_levels'], auxiliary['patient_code_adj'], auxiliary['code_code_adj']
    
    train_lab_x = tf.convert_to_tensor(train_lab_x, dtype=tf.float32)
    valid_lab_x = tf.convert_to_tensor(valid_lab_x, dtype=tf.float32)
    test_lab_x = tf.convert_to_tensor(test_lab_x, dtype=tf.float32)
    
    print(train_codes_x.shape, train_codes_y.shape, train_lab_x.shape, train_visit_lens.shape)
    print(valid_codes_x.shape, valid_codes_y.shape, valid_lab_x.shape, valid_visit_lens.shape)
    print(test_codes_x.shape, test_codes_y.shape, test_lab_x.shape, test_visit_lens.shape)
    
    
    
    pre_trained_model = tf.keras.models.load_model(pre_trained)
    weights = pre_trained_model.layers[0].get_weights()
    weight, bias = weights[0], weights[1]
    
    config = {
        'patient_code_adj': tf.constant(patient_code_adj, dtype=tf.float32),
        'code_code_adj': tf.constant(code_code_adj, dtype=tf.float32),
        'code_levels': tf.constant(code_levels, dtype=tf.int32),
        'code_num_in_levels': np.max(code_levels, axis=0) + 1,
        'patient_num': train_codes_x.shape[0],
        'max_visit_seq_len': train_codes_x.shape[1],
        'output_dim': len(code_map),
        'activation': None,
        'use_lab': use_lab, 
        'pre_trained_weight': weight,
        'pre_trained_bias': bias
    }

    test_historical = historical_hot(test_codes_x, len(code_map))
    
    
    visit_rnn_dims = [200]
    pre_trained_dims = 200
    hyper_params = {
        'code_dims': [32, 32, 32, 32],
        'patient_dim': 16,
        'patient_hidden_dims': [16],
        'code_hidden_dims': [64, 128],
        'visit_rnn_dims': visit_rnn_dims,
        'visit_attention_dim': 32, 
        'pre_trained_dims': pre_trained_dims,
        'dropout': 0.2
    }
    
    
    
    valid_codes_gen = DataGenerator([valid_codes_x, valid_visit_lens, valid_lab_x], shuffle=False)
    test_codes_gen = DataGenerator([test_codes_x, test_visit_lens, test_lab_x], shuffle=False)

    def lr_schedule_fn(epoch, lr):
        if epoch < 20:
            lr = 0.01
        elif epoch < 100:
            lr = 0.001
        elif epoch < 200:
            lr = 0.0001
        else:
            lr = 0.00001
        return lr


    lr_scheduler = LearningRateScheduler(lr_schedule_fn)
    valid_callback = EvaluateCodesCallBack(valid_codes_gen, valid_codes_y, historical=None)
    test_callback = EvaluateCodesCallBack(test_codes_gen, test_codes_y, historical=None)

    cgl_model = CGL(config, hyper_params)
    cgl_model.compile(optimizer='adam', loss=medical_codes_loss)
    cgl_model.fit(x={
        'visit_codes': train_codes_x,
        'visit_lens': train_visit_lens,
        'lab': train_lab_x
    }, y=train_codes_y.astype(float), epochs=100, batch_size=64, callbacks=[lr_scheduler, valid_callback], verbose = 2)
    cgl_model.summary()
    """validation_data=({
        'visit_codes': valid_codes_x,
        'visit_lens': valid_visit_lens,
        'word_ids': valid_note_x,
        'word_lens': valid_note_lens,
        'lab': valid_lab_x
    }, valid_codes_y.astype(float)),
    """
    