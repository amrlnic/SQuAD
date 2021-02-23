import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from ..layers import (WordEmbedding, CharEmbedding, HighwayNetwork, 
    CharCNN, ContextualEmbedding, Modelling, ModellingEnd, Start, End)
import pickle
import numpy as np
import json
import nltk
nltk.download('punkt')
from nltk import word_tokenize
import tensorflow.keras.backend as K

class BIDAF(Model):

  """
  the BIDAF model
  """

  def __init__(self, 
               question_maxlen, 
               context_maxlen, 
               word_vocab_len, 
               embedding_size, 
               embedding_matrix, 
               char_vocab_len,
               word_maxlen, 
               n_filters, 
               filter_size, 
               char_embedding_size,
               word_tokenizer_path,
               char_tokenizer_path,
               **kwargs):
    
    
    super(BIDAF, self).__init__(name = 'BIDAF', **kwargs)

    self.question_maxlen = question_maxlen
    self.context_maxlen = context_maxlen
    self.word_vocab_len = word_vocab_len
    self.embedding_size = embedding_size
    self.embedding_matrix = embedding_matrix
    self.char_vocab_len = char_vocab_len
    self.char_embedding_size = char_embedding_size
    self.word_maxlen = word_maxlen
    self.n_filters = n_filters
    self.filter_size = filter_size

    with open(word_tokenizer_path, 'rb') as handle:
      self.word_tokenizer = pickle.load(handle)

    with open(char_tokenizer_path, 'rb') as handle:
      self.char_tokenizer = pickle.load(handle)

    self.similarity_weights = Dense(1, use_bias = False)

    # layers
    self.word_embedding = WordEmbedding(self.word_vocab_len, self.embedding_size, self.question_maxlen, self.embedding_matrix)
    self.char_embedding = CharEmbedding(self.char_vocab_len, self.char_embedding_size, self.word_maxlen)
    self.cnn = CharCNN(self.n_filters, self.filter_size)
    self.highway = HighwayNetwork(hidden_size = self.embedding_size + self.n_filters)
    self.contextual = ContextualEmbedding(self.embedding_size)
    self.modelling = Modelling(self.embedding_size)
    self.modelling_end = ModellingEnd(self.embedding_size)
    self.output_start = Start()
    self.ouput_end = End()

  def _get_tokens(self):

    self.question = self.word_tokenizer.texts_to_sequences([self._question])
    self.context = self.word_tokenizer.texts_to_sequences([self._context])
    self.context_ids = self.context

  def _get_padded_sequences(self):

    self.question = tf.keras.preprocessing.sequence.pad_sequences(self.question, maxlen = self.question_maxlen, padding = 'post', truncating = 'post')
    self.context = tf.keras.preprocessing.sequence.pad_sequences(self.context, maxlen = self.context_maxlen, padding = 'post', truncating = 'post')

  def make_prediction(self, question, context):

    self._question = word_tokenize(question)
    self._context = word_tokenize(context)

    self._get_tokens()
    self._get_padded_sequences()

    self.__get_tokens()
    self.__get_padded_sequences()

    start, end = self.predict([
                      self.question,
                      self.context,
                      self.question_char,
                      self.context_char
                ])
    
    start = start.argmax()
    end = end.argmax() + 1

    if start > end:
      start = end
      end = start

    answer = ''

    for i in range(start, end):
      answer += self.word_tokenizer.index_word[self.context_ids[0][i]] + ' '
    return answer.strip()

  def multi_predictions(self, datasets, path):
    predictions = {}

    for dataset in datasets:

      for batch in dataset:

        if len(batch) == 3:
          sequences = batch[0]
          id = batch[2][0].tolist()
        else:
          sequences = batch[0]
          id = batch[1][0].tolist()          

        qw, cw, qc, cc = sequences

        start, end = self.predict([qw, cw, qc, cc])

        start = start.argmax(axis = 1)
        end = end.argmax(axis = 1)

        answers = []

        for idx, (s, e) in enumerate(zip(start, end)):
          if s > e:
            s = e
            e = s

          answer = ''
          for i in range(s,e):
            answer += self.word_tokenizer.index_word[cw[idx][i]] + ' '
          answers.append(answer.strip())
        
        predictions.update({i.strip(): a for i,a in zip(id, answers)})
    
    with open(path, 'w') as handle:
      json.dump(predictions, handle)

    print(f' the file containing the predictions has been created in {path}')
    
  def __get_tokens(self):

    self.question_char = self.char_tokenizer.texts_to_sequences(self._question)
    self.context_char = self.char_tokenizer.texts_to_sequences(self._context)

  def __get_padded_sequences(self):

    # pad question at the character level
    v = tf.keras.preprocessing.sequence.pad_sequences(self.question_char, padding = 'post', truncating = 'post', maxlen = self.word_maxlen)
    to_add = self.question_maxlen - v.shape[0]
    add = np.zeros((to_add, self.word_maxlen))
    arr = np.vstack([v,add])
    self.question_char = arr

    # pad context at the character level
    v = tf.keras.preprocessing.sequence.pad_sequences(self.context_char, padding = 'post', truncating = 'post', maxlen = self.word_maxlen)
    to_add = self.context_maxlen - v.shape[0]
    add = np.zeros((to_add, self.word_maxlen))
    arr = np.vstack([v,add])
    self.context_char = arr

    self.question_char = tf.expand_dims(self.question_char, axis = 0)
    self.context_char = tf.expand_dims(self.context_char, axis = 0)


  def call(self, inputs, training = True):
    qw, cw, qc, cc = inputs  # (bs, q_len), (bs, ctx_len), (bs, q_len, w_len), (bs, ctx_len, w_len)

    # embedding always non-trainable
    qw = self.word_embedding(qw) # (bs, q_len, emb)
    cw = self.word_embedding(cw) # (bs, ctx_len, emb)

    qc = self.char_embedding(qc) # (bs, q_len, w_len, char_emb)
    cc = self.char_embedding(cc) # (bs, ctx_len, w_len, char_emb)

    qc = self.cnn(qc) # (bs, q_len, n_filters)
    cc = self.cnn(cc) # (bs, ctx_len, n_filters)

    H = tf.concat([cw, cc], axis = 2) # (bs, ctx_len, emb + n_filters)
    U = tf.concat([qw, qc], axis = 2) # (bs, q_len, emb + n_filters)

    # highway
    H = self.highway(H) # (bs, ctx_len, emb + n_filters)
    U = self.highway(U) # (bs, q_len, emb + n_filters)

    # contextual embedding
    H = self.contextual(H) # (bs, ctx_len, emb + n_filters)
    U = self.contextual(U) # (bs, q_len, emb + n_filters)

    # similarity matrix
    expand_h = tf.concat([[1, 1], [tf.shape(U)[1]], [1]], axis = 0) # [1, 1, q_len, 1]
    expand_u = tf.concat([[1], [tf.shape(H)[1]], [1, 1]], axis = 0) # [1, ctx_len, 1, 1]

    h = tf.tile(tf.expand_dims(H, axis = 2), expand_h) # (bs, ctx_len, q_len, emb + n_filters)
    u = tf.tile(tf.expand_dims(U, axis = 1), expand_u) # (bs, ctx_len, q_len, emb + n_filters)
    h_u = h * u # (bs, ctx_len, q_len, emb + n_filters)

    alpha = tf.concat([h, u, h_u], axis = -1) # (bs, ctx_len, q_len, 3 * (emb + n_filters))
    
    similarity_matrix = self.similarity_weights(alpha) # (bs, ctx_len, q_len, 1)
    similarity_matrix = tf.squeeze(similarity_matrix, 3) # (bs, ctx_len, q_len)

    # context to query attention
    attention_weights = tf.nn.softmax(similarity_matrix, axis = -1) # (bs, ctx_len, q_len)
    C2Q = K.batch_dot(attention_weights, U) # (bs, ctx_len, emb + n_filters)

    # query to context attention
    attention_weights = tf.nn.softmax(tf.math.reduce_max(similarity_matrix, axis = 2), axis = -1) # (bs, ctx_len)
    attention_weights = tf.expand_dims(attention_weights, axis = 1) # (bs, 1, ctx_len)
    Q2C = K.batch_dot(attention_weights, H) # (bs, 1, emb + n_filters)
    Q2C = tf.tile(Q2C, [1, tf.shape(H)[1], 1]) # (bs, ctx_len, emb + n_filters)

    # query aware representation
    G = tf.concat([H, C2Q, (H * C2Q), (H * Q2C)], axis = 2) # (bs, ctx_len, 4 * (emb + n_filters) )

    # modelling
    M = self.modelling(G) # (bs, ctx_len, emb + n_filters)

    # output
    M2 = self.modelling_end([G,M]) # (bs, ctx_len, emb + n_filters)

    # start prediction
    start = self.output_start(tf.concat([G, M], axis = 2)) # (bs, ctx_len)

    # end prediction
    end = self.ouput_end(M2) # (bs, ctx_len)

    return start, end