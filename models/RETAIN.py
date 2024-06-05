import numpy as np
import tensorflow as tf
from tensorflow import keras


class RETAINLayer(tf.keras.layers.Layer):
    def __init__(self, unit, steps, input_dim=128):
        super(RETAINLayer, self).__init__()
        self.unit = unit  # omitted
        self.steps = steps  # omitted
        self.feature_size = input_dim  # feature_size == embedding dim
        self.dropout = 0.5
        self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout)

        self.alpha_gru = tf.keras.layers.GRU(self.feature_size, return_sequences=True)
        self.beta_gru = tf.keras.layers.GRU(self.feature_size, return_sequences=True)
        self.alpha_li = tf.keras.layers.Dense(1)
        self.beta_li = tf.keras.layers.Dense(self.feature_size)

    def reverse_x(self, x, lengths):
        reversed_input = tf.zeros_like(x)
        for i in range(tf.shape(x)[0]):
            length = lengths[i]
            reversed_seq = tf.reverse(x[i, :length], axis=[0])
            indices = tf.concat([tf.fill([length, 1], i), tf.range(length)[:, tf.newaxis]], axis=1)
            reversed_input = tf.tensor_scatter_nd_update(reversed_input, indices, reversed_seq)
        return reversed_input

    def compute_alpha(self, rx, lengths):
        mask = tf.sequence_mask(lengths, maxlen=tf.shape(rx)[1])
        g = self.alpha_gru(rx, mask=mask)
        attn_alpha = tf.nn.softmax(self.alpha_li(g), axis=1)
        return attn_alpha

    def compute_beta(self, rx, lengths):
        mask = tf.sequence_mask(lengths, maxlen=tf.shape(rx)[1])
        h = self.beta_gru(rx, mask=mask)
        attn_beta = tf.nn.tanh(self.beta_li(h))
        return attn_beta

    def call(self, x, mask):
        x = self.dropout_layer(x)
        batch_size = tf.shape(x)[0]
        if mask is None:
            lengths = tf.fill([batch_size], tf.shape(x)[1])
        else:
            lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
        rx = self.reverse_x(x, lengths)
        attn_alpha = self.compute_alpha(rx, lengths)
        attn_beta = self.compute_beta(rx, lengths)
        c = attn_alpha * attn_beta * x
        c = tf.reduce_sum(c, axis=1)
        return c


class Pre_trained(tf.keras.layers.Layer):
    def __init__(self, units, w, b):
        super().__init__()
        # self.dense1 = keras.layers.Dense(units, activation = None)
        self.dense2 = tf.keras.layers.Dense(units, kernel_initializer=tf.constant_initializer(w),
                                         bias_initializer=tf.constant_initializer(b), activation=None, trainable=False)

    def call(self, x, lab):
        lab = self.dense2(lab)
        output = tf.concat([x, lab], 1)
        return output


class Classifier(tf.keras.layers.Layer):
    def __init__(self, output_dim, name='classifier'):
        super().__init__(name=name)
        self.dense = tf.keras.layers.Dense(output_dim, activation=None)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x):
        x = self.dropout(x)
        output = self.dense(x)
        return output


class RETAIN(tf.keras.Model):
    def __init__(self, output_dim, w, b, steps, input_dim, use_lab, code_size):
        super().__init__()
        self.steps = steps
        self.embedding_dim = input_dim  # Embedding from encoded_dim -> embedding_dim
        self.output_dim = output_dim
        self.code_size = code_size
        self.use_lab = use_lab

        self.embedding = tf.keras.layers.Embedding(input_dim=self.code_size, output_dim=self.embedding_dim,
                                                   mask_zero=True)
        self.retain_layer = RETAINLayer(128, self.steps, self.embedding_dim)

        if self.use_lab:
            self.trained = Pre_trained(200, w, b)

        self.classifier = Classifier(output_dim)

    def call(self, inputs):
        x = self.embedding(inputs['codes_x'])  # (patient, visit, event, embedding_dim) - (32, 14, 100, 128)
        x = tf.reduce_sum(x, axis=2)  # (patient, visit, embedding_dim) - (32, 14, 128)
        mask = tf.reduce_sum(x, axis=2) != 0
        x = self.retain_layer(x, mask)  # (patient, embedding_dim) - (32, 128)
        if self.use_lab:
            x = self.trained(x, inputs['lab_x'])
        output = self.classifier(x)  # (patient, output_dim) - (32, 10)
        return output


if __name__ == '__main__':
    code_size = 1000
    embedding_dim = 128
    batch_size = 32
    sequence_length = 14
    feature_length = 100
    output_dim = 10

    w, b = None, None

    model = RETAIN(output_dim, w, b, sequence_length, embedding_dim, False, code_size)
    test_input = np.random.randint(0, code_size, size=(batch_size, sequence_length, feature_length))
    test_input = {'codes_x': tf.convert_to_tensor(test_input, dtype=tf.int32)}
    output = model(test_input)

    print("输入形状为：", test_input['codes_x'].shape)
    print("前向传播成功，输出形状为：", output.shape)
