import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import sklearn


def f1(y_true_hot, y_pred, metrics='weighted'):
    result = np.zeros_like(y_true_hot)
    for i in range(len(result)):
        true_number = np.sum(y_true_hot[i] == 1)
        result[i][y_pred[i][:true_number]] = 1
    return f1_score(y_true=y_true_hot, y_pred=result, average=metrics, zero_division=0)


def top_k_prec_recall(y_true_hot, y_pred, ks):
    a = np.zeros((len(ks), ))
    r = np.zeros((len(ks), ))
    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        for i, k in enumerate(ks):
            p = set(pred[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            r[i] += len(it) / len(t)
    return a / len(y_true_hot), r / len(y_true_hot)


def calculate_occurred(historical, y, preds, ks):
    r1 = np.zeros((len(ks), ))
    r2 = np.zeros((len(ks),))
    n = np.sum(y, axis=-1)
    for i, k in enumerate(ks):
        n_k = n
        pred_k = np.zeros_like(y)
        for T in range(len(pred_k)):
            pred_k[T][preds[T][:k]] = 1
        pred_occurred = np.logical_and(historical, pred_k)
        pred_not_occurred = np.logical_and(np.logical_not(historical), pred_k)
        pred_occurred_true = np.logical_and(pred_occurred, y)
        pred_not_occurred_true = np.logical_and(pred_not_occurred, y)
        r1[i] = np.mean(np.sum(pred_occurred_true, axis=-1) / n_k)
        r2[i] = np.mean(np.sum(pred_not_occurred_true, axis=-1) / n_k)
    return r1, r2


class EvaluateCodesCallBack1(Callback):
    def __init__(self, data_gen, y, historical=None, model_name="RNN", lab_binary=False, proc_binary=False):
        super().__init__()
        self.data_gen = data_gen
        self.y = y
        self.historical = historical

        self.model_name = model_name
        self.lab_binary = lab_binary
        self.proc_binary = proc_binary

        self.last_epoch = "None"
        self.last_f1_score = "None"
        self.last_recall = "None"

    def on_epoch_end(self, epoch, logs=None):
        step_size = len(self.data_gen)
        preds = []
        for i in range(step_size):
            if self.model_name in ['RNN', 'RETAIN']:
                if self.lab_binary:
                    batch_codes_x, batch_lab_x = self.data_gen[i]
                    inputs = {'codes_x': batch_codes_x, 'lab_x': batch_lab_x}
                else:
                    batch_codes_x = self.data_gen[i][0]
                    inputs = {'codes_x': batch_codes_x}
            elif self.model_name == 'CGL':
                batch_codes_x, batch_visit_lens, batch_lab_x = self.data_gen[i]
                inputs = {'visit_codes': batch_codes_x, 'visit_lens': batch_visit_lens, 'lab': batch_lab_x}
            else:
                ValueError('task must be either diagnosis_prediction or heart_failure_prediction')
                inputs = None
            output = self.model(inputs=inputs, training=False)
            logit = tf.math.sigmoid(output)
            pred = tf.argsort(logit, axis=-1, direction='DESCENDING')
            preds.append(pred.numpy())
        preds = np.vstack(preds)
        f1_score = f1(self.y, preds)
        prec, recall = top_k_prec_recall(self.y, preds, ks=[20, 40])
        if self.historical is not None:
            r1, r2 = calculate_occurred(self.historical, self.y, preds, ks=[10, 20, 30, 40])
            print('\t', 'f1_score:', f1_score, '\t', 'top_k_recall:', recall, '\t', 'occurred:', r1, '\t', 'not occurred:', r2)
        else:
            print('\t', 'f1_score:', f1_score, '\t', 'top_k_recall:', recall)

        self.last_epoch = epoch + 1
        self.last_f1_score = f1_score
        self.last_recall = recall

    def on_train_end(self, log=None):
        lab_prefix = '_lab' if self.lab_binary else ''
        proc_prefix = '_proc' if self.proc_binary else ''

        with open(f'./output/{self.model_name}{lab_prefix}{proc_prefix}.txt', 'a') as f:
            f.write(f'# {self.last_epoch} # {self.last_f1_score} # {self.last_recall}\n')
            f.write('--\n')


class EvaluateCodesCallBackRNNOnly(Callback):
    def __init__(self, data_gen, y, model_name, lab_binary=False, proc_binary=False):
        super().__init__()
        self.data_gen = data_gen
        self.y = y

        self.model_name = model_name
        self.lab_binary = lab_binary
        self.proc_binary = proc_binary

        self.last_epoch = "None"
        self.last_f1_score = "None"
        self.last_recall = "None"

    def on_epoch_end(self, epoch, logs=None):
        step_size = len(self.data_gen)
        preds = []
        for i in range(step_size):
            batch_codes_x = self.data_gen[i][0]
            output = self.model(inputs={
                'codes_x': batch_codes_x
            }, training=False)
            logits = tf.math.sigmoid(output)
            pred = tf.argsort(logits, axis=-1, direction='DESCENDING')
            preds.append(pred.numpy())
        preds = np.vstack(preds)
        f1_score = f1(self.y, preds)
        prec, recall = top_k_prec_recall(self.y, preds, ks=[20, 40])
        print('\t', 'f1_score:', f1_score, '\t', 'top_k_recall:', recall)

        self.last_epoch = epoch + 1
        self.last_f1_score = f1_score
        self.last_recall = recall

    def on_train_end(self, log=None):
        lab_prefix = '_lab' if self.lab_binary else ''
        proc_prefix = '_proc' if self.proc_binary else ''

        with open(f'./output/{self.model_name}{lab_prefix}{proc_prefix}.txt', 'a') as f:
            f.write(f'# {self.last_epoch} # {self.last_f1_score} # {self.last_recall}\n')
            f.write('--\n')


class EvaluateHFCallBack(Callback):
    def __init__(self, data_gen, y, historical=None, model_name="RNN", lab_binary=False, proc_binary=False):
        super().__init__()
        self.data_gen = data_gen
        self.y = y
        self.historical = historical

        self.model_name = model_name
        self.lab_binary = lab_binary
        self.proc_binary = proc_binary

        self.last_epoch = "None"
        self.last_f1_score = "None"
        self.last_auc = "None"

    def on_epoch_end(self, epoch, logs=None):
        step_size = len(self.data_gen)
        preds, outputs = [], []
        for i in range(step_size):
            if self.model_name in ['RNN', 'RETAIN']:
                if self.lab_binary:
                    batch_codes_x, batch_lab_x = self.data_gen[i]
                    inputs = {'codes_x': batch_codes_x, 'lab_x': batch_lab_x}
                else:
                    batch_codes_x = self.data_gen[i][0]
                    inputs = {'codes_x': batch_codes_x}
            elif self.model_name == 'CGL':
                batch_codes_x, batch_visit_lens, batch_lab_x = self.data_gen[i]
                inputs = {'visit_codes': batch_codes_x, 'visit_lens': batch_visit_lens, 'lab': batch_lab_x}
            else:
                inputs = None
            output = self.model(inputs=inputs, training=False)
            outputs.append(tf.squeeze(output).numpy())
            pred = tf.squeeze(tf.cast(output > 0.5, tf.int32))
            preds.append(pred.numpy())
        outputs = np.concatenate(outputs)
        preds = np.concatenate(preds)
        auc = roc_auc_score(self.y, outputs)
        f1_score = sklearn.metrics.f1_score(self.y, preds)
        print('\t', 'auc:', auc, '\t', 'f1_score:', f1_score)

        self.last_epoch = epoch + 1
        self.last_f1_score = f1_score
        self.last_auc = auc

    def on_train_end(self, log=None):
        lab_prefix = '_lab' if self.lab_binary else ''
        proc_prefix = '_proc' if self.proc_binary else ''

        with open(f'./output/{self.model_name}_hf{lab_prefix}{proc_prefix}.txt', 'a') as f:
            f.write(f'# {self.last_epoch} # {self.last_f1_score} # {self.last_auc}\n')
            f.write('--\n')


# class EvaluateCodesCallBackRNN(Callback):
#     def __init__(self, data_gen, y):
#         super().__init__()
#         self.data_gen = data_gen
#         self.y = y
#
#     def on_epoch_end(self, epoch, logs=None):
#         step_size = len(self.data_gen)
#         preds = []
#         for i in range(step_size):
#             batch_codes_x, batch_lab_x = self.data_gen[i]
#             output = self.model(inputs={
#                 'codes_x': batch_codes_x,
#                 'lab_x': batch_lab_x
#             }, training=False)
#             logits = tf.math.sigmoid(output)
#             pred = tf.argsort(logits, axis=-1, direction='DESCENDING')
#             preds.append(pred.numpy())
#         preds = np.vstack(preds)
#         f1_score = f1(self.y, preds)
#         prec, recall = top_k_prec_recall(self.y, preds, ks=[20, 40])
#         print('\t', 'f1_score:', f1_score, '\t', 'top_k_recall:', recall)


# class EvaluateCodesCallBack(Callback):
#     def __init__(self, data_gen, y, historical=None):
#         super().__init__()
#         self.data_gen = data_gen
#         self.y = y
#         self.historical = historical
#
#     def on_epoch_end(self, epoch, logs=None):
#         step_size = len(self.data_gen)
#         preds = []
#         for i in range(step_size):
#             batch_codes_x, batch_visit_lens, batch_lab_x = self.data_gen[i]
#             output = self.model(inputs={
#                 'visit_codes': batch_codes_x,
#                 'visit_lens': batch_visit_lens,
#                 'lab': batch_lab_x
#             }, training=False)
#             logits = tf.math.sigmoid(output)
#             pred = tf.argsort(logits, axis=-1, direction='DESCENDING')
#             preds.append(pred.numpy())
#         preds = np.vstack(preds)
#         f1_score = f1(self.y, preds)
#         prec, recall = top_k_prec_recall(self.y, preds, ks=[10, 20, 30, 40])
#         if self.historical is not None:
#             r1, r2 = calculate_occurred(self.historical, self.y, preds, ks=[10, 20, 30, 40])
#             print('\t', 'f1_score:', f1_score, '\t', 'top_k_recall:', recall, '\t', 'occurred:', r1, '\t', 'not occurred:', r2)
#         else:
#             print('\t', 'f1_score:', f1_score, '\t', 'top_k_recall:', recall)
