import tensorflow as tf
import pandas as pd
import numpy as np
import os
import pickle

class SQUAD_dataset(tf.keras.utils.Sequence):

  def __init__(self, data, question_maxlen, context_maxlen, word_maxlen, batch_size, with_answer = True):
    self.QUESTION_MAXLEN = question_maxlen
    self.CONTEXT_MAXLEN = context_maxlen
    self.WORD_MAXLEN = word_maxlen
    self.batch_size = batch_size
    self.with_answer = with_answer
    self.__get_batches(data)

  def __len__(self):
    return len(self.batches)

  def __get_batches(self, data):
    batches = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
    self.batches = batches

  def __repr__(self):
    template = '''SQUAD_dataset : questions : ({0}, {1}), contexts : ({0}, {2}), char_questions : ({0}, {1}, {3}), char_contexts : ({0}, {2}, {3}), index : ({0}, 1)'''.format(self.batch_size, self.QUESTION_MAXLEN, self.CONTEXT_MAXLEN, self.WORD_MAXLEN)
    return template

  @classmethod
  def from_file(cls, path):
    path = os.path.join(os.getcwd(), path)
    with open(path, 'rb') as handle:
      dataset = pickle.load(handle)
    return dataset

  def to_pickle(self, path):
    path = os.path.join(os.getcwd(), path)
    folder = os.path.dirname(path)

    if not os.path.exists(folder):
      os.makedirs(folder)

    with open(path, 'wb') as handle:
      pickle.dump(self, handle, protocol = pickle.HIGHEST_PROTOCOL)

  def __getitem__(self, idx):
    batch = self.batches[idx].reset_index(drop = True)
    
    id = np.asarray(batch['id'])

    # questions and contexts words padding
    q_w = tf.keras.preprocessing.sequence.pad_sequences(batch['tokenized_question'], padding = 'post', maxlen = self.QUESTION_MAXLEN)
    c_w = tf.keras.preprocessing.sequence.pad_sequences(batch['tokenized_context'], padding = 'post', maxlen = self.CONTEXT_MAXLEN)

    # question_char padding
    q_c = np.zeros((q_w.shape[0], self.QUESTION_MAXLEN, self.WORD_MAXLEN), dtype = np.int32)

    for i, value in batch['char_tokenized_question'].iteritems():
      v = tf.keras.preprocessing.sequence.pad_sequences(value, padding = 'post', maxlen = self.WORD_MAXLEN, truncating = 'post')
      to_add = self.QUESTION_MAXLEN - v.shape[0]
      add = np.zeros((to_add, self.WORD_MAXLEN))
      arr = np.vstack([v,add])
      q_c[i] = arr

    # context_chr padding
    c_c = np.zeros((q_w.shape[0], self.CONTEXT_MAXLEN, self.WORD_MAXLEN), dtype = np.int32)

    for i, value in batch['char_tokenized_context'].iteritems():
      v = tf.keras.preprocessing.sequence.pad_sequences(value, padding = 'post', maxlen = self.WORD_MAXLEN, truncating = 'post')
      to_add = self.CONTEXT_MAXLEN - v.shape[0]
      add = np.zeros((to_add, self.WORD_MAXLEN))
      arr = np.vstack([v,add])
      c_c[i] = arr

    if self.with_answer:
        # one hot encode start and end
        y_start = tf.keras.utils.to_categorical(batch['start'].values, self.CONTEXT_MAXLEN)
        y_end = tf.keras.utils.to_categorical(batch['end'].values, self.CONTEXT_MAXLEN)

        # (inputs), (outputs), (id,)
        return (q_w, c_w, q_c, c_c), (y_start, y_end), (id,)
    return (q_w, c_w, q_c, c_c), (id, )