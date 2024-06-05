import os
import _pickle as pickle

from metrics import *
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import numpy as np
from utils import DataGenerator
from metrics import EvaluateCodesCallBack1
from loss import medical_codes_loss
from models.GRU import GRU


mimic3_path = os.path.join('data', 'mimic3')
standard_path = os.path.join(mimic3_path, 'standard')
pre_trained = os.path.join(mimic3_path, 'Pre-trained')


def load_data():
    return pickle.load(open(os.path.join(standard_path, 'codes_dataset.pkl'), 'rb'))


def load_hf_data():
    return pickle.load(open(os.path.join(standard_path, 'heart_failure.pkl'), 'rb'))


def convert_hot(x):
    result = np.zeros_like(x ,dtype=int)
    for i, examples in enumerate(x):
        for j, step in enumerate(examples):
            if np.all(x[i][j]==0): break
            hot_index = [t-1 for t in x[i][j] if t!= 0]
            result[i][j][hot_index] = 1
    return result


def lr_schedule_fn(epoch, lr):
    if epoch < 10:
        lr = 0.01
    elif epoch < 50:
        lr = 0.001
    elif epoch < 80:
        lr = 0.0001
    else:
        lr = 0.00005
    return lr


if __name__ == '__main__':
    task = 'heart_failure_prediction'  # ['diagnosis_prediction', 'heart_failure_prediction']
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    codes_dataset = load_data()
    train_codes_data, valid_codes_data, test_codes_data = codes_dataset['train_codes_data'], codes_dataset['valid_codes_data'], codes_dataset['test_codes_data']

    if task == 'diagnosis_prediction':
        (train_codes_x, train_codes_y, train_lab_x, train_visit_lens) = train_codes_data  # , train_proc_x
        (valid_codes_x, valid_codes_y, valid_lab_x, valid_visit_lens) = valid_codes_data  # , valid_proc_x
        (test_codes_x, test_codes_y, test_lab_x, test_visit_lens) = test_codes_data  # , test_proc_x
    elif task == 'heart_failure_prediction':
        (train_codes_x, _, train_lab_x, train_visit_lens) = train_codes_data
        (valid_codes_x, _, valid_lab_x, valid_visit_lens) = valid_codes_data
        (test_codes_x, _, test_lab_x, test_visit_lens) = test_codes_data
        heart_failure = load_hf_data()
        train_codes_y, valid_codes_y, test_codes_y = heart_failure['train_hf_y'], heart_failure['valid_hf_y'], heart_failure['test_hf_y']
        train_codes_y, valid_codes_y, test_codes_y = train_codes_y.reshape((-1, 1)), valid_codes_y.reshape((-1, 1)), test_codes_y.reshape((-1, 1))
    else:
        raise ValueError('task must be either diagnosis_prediction or heart_failure_prediction')

    train_codes_x, valid_codes_x, test_codes_x = convert_hot(train_codes_x), convert_hot(valid_codes_x), convert_hot(test_codes_x)

    train_codes_x = tf.convert_to_tensor(train_codes_x, dtype=tf.float32)
    train_codes_y = tf.convert_to_tensor(train_codes_y, dtype=tf.float32)
    valid_codes_x = tf.convert_to_tensor(valid_codes_x, dtype=tf.float32)
    valid_codes_y = tf.convert_to_tensor(valid_codes_y, dtype=tf.float32)
    test_codes_x = tf.convert_to_tensor(test_codes_x, dtype=tf.float32)
    test_codes_y = tf.convert_to_tensor(test_codes_y, dtype=tf.float32)
    train_lab_x = tf.convert_to_tensor(train_lab_x, dtype=tf.float32)
    valid_lab_x = tf.convert_to_tensor(valid_lab_x, dtype=tf.float32)
    test_lab_x = tf.convert_to_tensor(test_lab_x, dtype=tf.float32)

    print(train_codes_x.shape, valid_codes_x.shape, test_codes_x.shape)
    print(train_codes_y.shape, valid_codes_y.shape, test_codes_y.shape)

    test_codes_gen = DataGenerator([test_codes_x, test_lab_x], shuffle=False)

    max_admission = len(train_codes_x[0])
    in_dim = len(train_codes_x[0][0])
    out_dim = len(train_codes_y[0])
    pre_trained_model = tf.keras.models.load_model(pre_trained)
    weights = pre_trained_model.layers[0].get_weights()
    weight, bias = weights[0], weights[1]
    lr_scheduler = LearningRateScheduler(lr_schedule_fn)

    for use_lab in [True]:  # , False
        for i in range(50):
            print('use_lab:', use_lab)
            print('iteration:', i+1)

            if task == 'diagnosis_prediction':
                test_callback = EvaluateCodesCallBack1(test_codes_gen, test_codes_y, model_name='RNN',
                                                       lab_binary=use_lab, proc_binary=False)
            elif task == 'heart_failure_prediction':
                test_callback = EvaluateHFCallBack(test_codes_gen, test_codes_y, model_name="RNN",
                                                   lab_binary=use_lab, proc_binary=False)
            else:
                raise ValueError('task must be either diagnosis_prediction or heart_failure_prediction')

            model = GRU(out_dim, weight, bias, max_admission, in_dim, use_lab)

            if task == 'diagnosis_prediction':
                model.compile(optimizer='adam', loss=medical_codes_loss)
            elif task == 'heart_failure_prediction':
                model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[tfa.metrics.F1Score(num_classes=1)])
            else:
                raise ValueError('task must be either diagnosis_prediction or heart_failure_prediction')

            model.fit(x={'codes_x': train_codes_x, 'lab_x': train_lab_x}, y=train_codes_y,
                      batch_size=128, epochs=100, callbacks=[test_callback, lr_scheduler], verbose=2)
            model.summary()
