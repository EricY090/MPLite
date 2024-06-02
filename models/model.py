import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Concatenate

from models.layers import HierarchicalEmbedding, PatientEmbedding
from models.layers import GraphConvolution
from models.layers import VisitEmbedding, TemporalEmbedding
from models.layers import Pre_trained

class CGLFeatureExtractor(Layer):
    def __init__(self, config, hyper_params, name='cgl_feature'):
        super().__init__(name=name)
        self.config = config
        self.hyper_params = hyper_params
        self.hierarchical_embedding_layer = HierarchicalEmbedding(
            code_levels=config['code_levels'],
            code_num_in_levels=config['code_num_in_levels'],
            code_dims=hyper_params['code_dims'])
        self.patient_embedding_layer = PatientEmbedding(
            patient_num=config['patient_num'],
            patient_dim=hyper_params['patient_dim'])
        self.graph_convolution_layer = GraphConvolution(
            patient_dim=hyper_params['patient_dim'],
            code_dim=np.sum(hyper_params['code_dims']),
            patient_code_adj=config['patient_code_adj'],
            code_code_adj=config['code_code_adj'],
            patient_hidden_dims=hyper_params['patient_hidden_dims'],
            code_hidden_dims=hyper_params['code_hidden_dims'])
        self.visit_embedding_layer = VisitEmbedding(
            max_seq_len=config['max_visit_seq_len'])
        self.visit_temporal_embedding_layer = TemporalEmbedding(
            rnn_dims=hyper_params['visit_rnn_dims'],
            attention_dim=hyper_params['visit_attention_dim'],
            max_seq_len=config['max_visit_seq_len'],
            name='visit_temporal')
        ### Code added here
        if config['use_lab']:
            self.pre_trained_layer = Pre_trained(
                hyper_params['pre_trained_dims'],
                config['pre_trained_weight'],
                config['pre_trained_bias'])
        ###
        

    def call(self, inputs, training=True):
        visit_codes = inputs['visit_codes']  # (batch_size, max_seq_len, max_code_num_in_a_visit)
        visit_lens = tf.reshape(inputs['visit_lens'], (-1, ))  # (batch_size, )
        
        code_embeddings = self.hierarchical_embedding_layer(None)
        patient_embddings = self.patient_embedding_layer(None)

        patient_embddings, code_embeddings = self.graph_convolution_layer(
            patient_embeddings=patient_embddings, code_embeddings=code_embeddings)
        visits_embeddings = self.visit_embedding_layer(
            code_embeddings=code_embeddings,
            visit_codes=visit_codes,
            visit_lens=visit_lens)
        visit_output, alpha_visit = self.visit_temporal_embedding_layer(visits_embeddings, visit_lens)
        ### Codes added here
        if self.config['use_lab']:
            visit_output = self.pre_trained_layer(visit_output, inputs['lab'])
        ###
        output = visit_output

        return output


class Classifier(Layer):
    def __init__(self, output_dim, dropout, activation=None, name='classifier'):
        super().__init__(name=name)
        self.dense = Dense(output_dim, activation=activation)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        x = self.dropout(x)
        output = self.dense(x)

        return output


class CGL(Model):
    def __init__(self, config, hyper_params, name='cgl'):
        super().__init__(name=name)
        self.cgl_feature_extractor = CGLFeatureExtractor(config, hyper_params)
        self.classifier = Classifier(config['output_dim'], hyper_params['dropout'], activation=config['activation'])

    def call(self, inputs, training=True):
        output = self.cgl_feature_extractor(inputs, training=training)
        output = self.classifier(output)
        return output
