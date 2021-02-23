from tensorflow.keras.layers import (Embedding, Layer, Conv1D, 
    TimeDistributed, GlobalMaxPooling1D, 
    Dense, Bidirectional, LSTM, Dropout)
    
import tensorflow as tf

class WordEmbedding(Layer):
    
    def __init__(self, input_dim, output_dim, input_len, embedding_matrix, trainable = False, mask_zero = True, **kwargs):
        
        super(WordEmbedding, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_len = input_len
        self.embedding_matrix = embedding_matrix
        self.trainable = trainable
        self.mask_zero = mask_zero

        self.word_embed = Embedding(
            input_dim = self.input_dim,
            output_dim = self.output_dim,
            weights = [self.embedding_matrix],
            trainable = self.trainable,
            input_length = self.input_len,
            mask_zero = self.mask_zero,
        )

    def build(self, input_shape):
      self.built = True

    def call(self, inputs):
        input = inputs
        return self.word_embed(input) 
    
    # inplement this method in order to get a serializable layer as part of a Functional model
    def get_config(self):
        # the base Layer class takes some keywords arguments like name and dtype, it is good to include 
        # them in the config (so we call the parent method and use the update method)
        config = super().get_config().copy()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'input_len': self.input_len, 
            'trainable': self.trainable,
            'mask_zero': self.mask_zero
        })
        return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)
      
      
class CharEmbedding(Layer):
    
    def __init__(self, input_dim, output_dim, input_len, **kwargs):
        
        super(CharEmbedding, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_len = input_len
        self.char_embed = Embedding(
            input_dim = self.input_dim, 
            output_dim = self.output_dim,  
            input_length = self.input_len
        )
        # This wrapper allows to apply a layer to every temporal slice of an input.
        # so we apply the same Embedding to every timestep (index 1) independently
        self.timed = TimeDistributed(self.char_embed)
        

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        return self.timed(inputs)
            
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'input_len': self.input_len, 
        })
        return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)
      
class CharCNN(Layer):
    
    def __init__(self, n_filters, filter_width, **kwargs):
        
        super(CharCNN, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.conv = Conv1D(self.n_filters, self.filter_width)
        self.pool = GlobalMaxPooling1D()
        self.timed = TimeDistributed(self.pool)
          
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        return self.timed(self.conv(inputs))
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_filters': self.n_filters,
            'filter_width': self.filter_width, 
        })
        return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)
      
class HighwayNetwork(Layer):
    
    def __init__(self, hidden_size, **kwargs):
        
        super(HighwayNetwork, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.normal = Dense(self.hidden_size, activation = 'relu') 
        self.transform_gate = Dense(self.hidden_size, activation = 'sigmoid')
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):        
        
        n = self.normal(inputs)
        g = self.transform_gate(inputs)
        x = g*n + (1-g)*inputs 
        return x

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'hidden_size': self.hidden_size, 
        })
        return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)
      
class ContextualEmbedding(Layer):
    
    def __init__(self, output_dim, **kwargs):
        
        super(ContextualEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.contextual = Bidirectional(LSTM(self.output_dim, return_sequences = True, dropout = 0.2))

    def build(self, input_shape):
        self.built = True 

    def call(self, inputs):
        return self.contextual(inputs)
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
      return cls(**config)
      
class Modelling(Layer):
    
    def __init__(self, output_dim, **kwargs):
        
        super(Modelling, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.modelling1 = Bidirectional(LSTM(self.output_dim, return_sequences = True, dropout = 0.2))
        self.modelling2 = Bidirectional(LSTM(self.output_dim, return_sequences = True, dropout = 0.2))
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        return self.modelling2(self.modelling1(inputs))
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)
      
class Start(Layer):
    
    def __init__(self, **kwargs):
        
        super(Start, self).__init__(**kwargs)
        self.dense = Dense(1, activation = 'linear', use_bias = False)
        self.dropout = Dropout(0.2)
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        
        GM = inputs
        start = self.dense(GM)
        start = self.dropout(start)
        p1 = tf.nn.softmax(tf.squeeze(start, axis = 2))
        return p1

    def get_config(self):
      
      config = super().get_config().copy()
      return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)
      
class ModellingEnd(Layer):
    
    def __init__(self, output_dim, **kwargs):
        
        super(ModellingEnd, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.end = Bidirectional(LSTM(self.output_dim, return_sequences = True, dropout = 0.2))
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        
        G, M = inputs
        M2 = self.end(M)
        GM2 = tf.concat([G, M2], axis = 2)
        return GM2
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)
      
class End(Layer):
    
    def __init__(self, **kwargs):
        
        super(End, self).__init__(**kwargs)
        self.dense = Dense(1, activation = 'linear', use_bias = False)
        self.dropout = Dropout(0.2)
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        
        GM2 = inputs
        end = self.dense(GM2)
        end = self.dropout(end)
        p2 = tf.nn.softmax(tf.squeeze(end, axis = 2))
        
        return p2


    def get_config(self):

      config = super().get_config().copy()

      return config
    
    @classmethod
    def from_config(cls, config):
      return cls(**config)

