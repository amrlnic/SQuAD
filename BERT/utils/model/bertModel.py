from transformers import BertModel, BertConfig
import os, re, json, requests, io, string
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig
import pandas as pd
import numpy as np
import pickle
import random

class BERT(TFBertModel):

    def __init__(
        self,
        enc_dec=True,
        enc_dim=128,
        dec_dim=64,
        rec_mod='biLSTM',
        bert_ft=True,
        dropout=False,
        drop_prob=0.5
    ):
        # Initializing a BERT bert-base-uncased style configuration
        configuration = BertConfig()

        """ 
        Returns a keras model for predicting the start and the end of the answer

        enc_dec (boolean): whether to use the encoder decoder model or not. If False, the base model will be used
        enc_dim (int): encoding dimension
        dec_dim (int): decoding dimension
        rec_mod (string): type of recurrent modules // 'biLSTM' or 'GRU'
        bert_ft (boolean): whether or not the bert will be fine - tuned
        dropout (boolean): whether or not using the dropout
        drop_prob (double): dropout probability
        """

        # use pre - trained BERT for creating the embeddings
        bert_model = TFBertModel.from_pretrained("bert-base-uncased")
        if not bert_ft:
            for layer in bert_model.layers:
                layer.trainable = False

        # input
        input_ids = layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
        token_type_ids = layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
        attention_mask = layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
        embeddings = bert_model(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )[0]

        if enc_dec:  # model with encoder - decoder

            if rec_mod == 'biLSTM':

                encoder = layers.Bidirectional(layers.LSTM(enc_dim, return_sequences=True),
                                            merge_mode='concat')(embeddings)

                decoder = layers.Bidirectional(layers.LSTM(dec_dim, return_sequences=True),
                                            merge_mode='concat')(encoder)

                high_dim = dec_dim*2  # number of units of the dense layers of the highway network

            else:

                encoder = layers.GRU(enc_dim, return_sequences=True)(embeddings)

                decoder = layers.GRU(dec_dim, return_sequences=True)(encoder)

                high_dim = dec_dim

            # highway network
            x_proj = layers.Dense(units=high_dim, activation='relu')(decoder)
            x_gate = layers.Dense(units=high_dim, activation='sigmoid')(decoder)

            x = (x_proj * x_gate) + (1 - x_gate) * decoder

        else:  # base model

            x = embeddings

        # dropout
        if dropout:
            x = layers.Dropout(drop_prob)(x)

        # output

        start_logits = layers.Dense(1, use_bias=False)(x)
        start_logits = layers.Flatten()(start_logits)

        end_logits = layers.Dense(1, use_bias=False)(x)
        end_logits = layers.Flatten()(end_logits)

        start_probs = layers.Activation(keras.activations.softmax)(start_logits)
        end_probs = layers.Activation(keras.activations.softmax)(end_logits)

        model = keras.Model(
            inputs=[input_ids, token_type_ids, attention_mask],
            outputs=[start_probs, end_probs]
        )

        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        optimizer = keras.optimizers.Adam(lr=5e-5)
        model.compile(optimizer=optimizer, loss=[loss, loss])
