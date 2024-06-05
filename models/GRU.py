import tensorflow as tf
from tensorflow import keras


class GRULayer(tf.keras.layers.Layer):
    def __init__(self, unit, steps, input_dim):
        super().__init__()
        self.block0 = tf.keras.layers.Masking(mask_value=0, input_shape=(steps, input_dim))
        self.block1 = tf.keras.layers.GRU(unit)  ###kernel_regularizer=tf.keras.regularizers.l2(0.01)

    def call(self, codes):
        x = self.block0(codes)
        x = self.block1(x)
        return x


class Pre_trained(tf.keras.layers.Layer):
    def __init__(self, units, w, b):
        super().__init__()
        # self.dense1 = keras.layers.Dense(units, activation = None)
        self.dense2 = keras.layers.Dense(units, kernel_initializer=tf.constant_initializer(w),
                                         bias_initializer=tf.constant_initializer(b), activation=None, trainable=False)

    def call(self, x, lab):
        # x = self.dense1(x)
        lab = self.dense2(lab)
        output = tf.concat([x, lab], 1)
        # output = tf.nn.relu(output)
        return output


class Classifier(tf.keras.layers.Layer):
    def __init__(self, output_dim, name='classifier'):
        super().__init__(name=name)
        self.dense = keras.layers.Dense(output_dim, activation=None)
        self.dropout = tf.keras.layers.Dropout(0.4)

    def call(self, x):
        x = self.dropout(x)
        output = self.dense(x)
        return output


class GRU(tf.keras.Model):
    def __init__(self, output_dim, w, b, steps, input_dim, use_lab):
        super().__init__()
        self.use_lab = use_lab
        self.gru = GRULayer(128, steps, input_dim)
        if self.use_lab:
            self.trained = Pre_trained(200, w, b)
        self.classifier = Classifier(output_dim)

    def call(self, inputs):
        x = self.gru(inputs['codes_x'])
        if self.use_lab:
            x = self.trained(x, inputs['lab_x'])
        output = self.classifier(x)
        return output