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
import json

def create_model(enc_dec, enc_dim, dec_dim,
                 rec_mod, bert_ft,
                 dropout, drop_prob, text_max_len):
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
    input_ids = layers.Input(shape=(text_max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(text_max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(text_max_len,), dtype=tf.int32)
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

    return model

def predict(model, x_eval, path, dataset, tokenizer):
  """  
  The model outputs are the logits: we take them to find the answer position in the context
  """
  
  raw_predictions = model.predict(x_eval) 

  predictions = {}
  for i in range(len(raw_predictions[0])):
    start=np.argmax(raw_predictions[0][i])
    end=np.argmax(raw_predictions[1][i])
    tokenized_answer = x_eval[0][i:i+1][0][start:end+1]

    decoded = tokenizer.decode(tokenized_answer)

    predictions[dataset.iloc[i]['index']] = decoded

  ##### Save model predictions on val set as a .JSON file  #####
  with open(path, 'w') as fp:
      json.dump(predictions, fp)