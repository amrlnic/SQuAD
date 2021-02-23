import json
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
from nltk import word_tokenize
nltk.download('punkt')
import gensim.downloader as gloader
from sklearn.model_selection import train_test_split
import re
import pickle

def load_dataset(path, record_path = ['data', 'paragraphs', 'qas', 'answers'], verbose = True, with_answer = True):

  """
  parse the SQUAD dataset into a dataframe
  """

  if verbose:
      print("Reading the json file")
  # if the file encoding is not UTF8 an exception should be raised    
  file = json.loads(open(path).read())

  if verbose:
      print("[INFO] processing...")

  # parsing different level's in the json file
  if with_answer:
    js = pd.json_normalize(file , record_path )
  m = pd.json_normalize(file, record_path[:-1] )
  r = pd.json_normalize(file, record_path[:-2])
  title = pd.json_normalize(file['data'], record_path = ['paragraphs'], meta = 'title')
  t = pd.json_normalize(file, record_path[0])

  #combining it into single dataframe
  contexts = np.repeat(r['context'].values, r['qas'].str.len())
  m['context'] = contexts
  m['title'] = np.repeat(title['title'].values, r['qas'].str.len())
  m['c_id'] = m['context'].factorize()[0]
  m = m.drop(['answers'], axis = 1)

  if with_answer:
    main = js.merge(m, left_index = True, right_index = True)
  else:
    main = m
  if verbose:
      print(f"[INFO] there are {main.shape[0]} questions with single answer")
      print(f"[INFO] there are {main.groupby('c_id').sum().shape[0]} different contexts")
      print(f"[INFO] there are {len(t)} unrelated subjects")
      print("[INFO] Done")
  return main

def download_glove_model(embedding_dimension = 50):

  """
  download glove model
  """

  download_path = "glove-wiki-gigaword-{}".format(embedding_dimension)
  try:
    print('[INFO] downloading glove {}'.format(embedding_dimension))
    emb_model = gloader.load(download_path)
    print('[INFO] done !')
  except ValueError as e:
      print("Glove: 50, 100, 200, 300")
      raise e
  return emb_model

def preprocess_sentence(text):

  """
  lowercase and strip the given text
  """

  text = text.lower()
  text = text.strip()
  return text


def clean_dataset(dataset, with_answer = True):

  """
  preprocess the dataset
  """

  _dataset = dataset.copy()

  cleaned_questions = _dataset['question'].apply(preprocess_sentence)

  # we process only different contexts and then we duplicate them
  unique_context = pd.Series(_dataset['context'].unique())
  count_c = _dataset.groupby('c_id').size()
  cleaned_contexts = unique_context.apply(preprocess_sentence)

  _dataset['question'] = cleaned_questions

  if with_answer:
    cleaned_texts = _dataset['text'].apply(preprocess_sentence)
    _dataset['text'] = cleaned_texts
  _dataset['context'] = pd.Series(np.repeat(cleaned_contexts, count_c).tolist())

  return _dataset


def get_tokenizer(dataset, glove_model = None):

  """
  create the word and char tokenizers and feed them 
  on the given dataset and the glove vocabulary
  """

  tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token = 'UNK', filters = '')

  # we will only keep the 200 - 1 most frequent characters (otherwise oom issue)
  # others tokens are replaced by UNK token 
  # we keep 199 most frequent tokens and indice 1 is UNK token (so we keep 198 tokens)
  char_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level = True, filters = '', oov_token = 'UNK', num_words = 200)

  if glove_model == None:
    glove_model = download_glove_model(EMBEDDING_SIZE)

  tokenized_questions = dataset['question'].apply(word_tokenize).to_list()

  contexts = pd.Series(dataset['context'].unique())
  tokenized_contexts = contexts.apply(word_tokenize).to_list()

  sequences = glove_model.index2entity + tokenized_questions + tokenized_contexts

  del glove_model # we  don't need anymore the glove model

  tokenizer.fit_on_texts(sequences)
  char_tokenizer.fit_on_texts(dataset['question'].to_list() + contexts.to_list())

  return tokenizer, char_tokenizer


def update_tokenizer(dataset, tokenizer, char_tokenizer):

  """
  update the existing word/char vocabulary on a new dataset
  """

  tokenized_questions = dataset['question'].apply(word_tokenize).to_list()

  contexts = pd.Series(dataset['context'].unique())
  tokenized_contexts = contexts.apply(word_tokenize).to_list()

  sequences = tokenized_questions + tokenized_contexts
  tokenizer.fit_on_texts(sequences)

  char_tokenizer.fit_on_texts(dataset['question'].to_list() + contexts.to_list())

def get_start_end(row):

  """
  get the start and end span for each sample,
  if the span cannot be found return -1
  """

  context = row['context']
  answer = row['text']
  tok_answer = word_tokenize(answer)

  _start = context.find(answer)

  if _start == -1:
    # the answer is not in the context
    # maybe due to a typo
    row['start'] = -1
    row['end'] = -1
    return row

  lc = context[:_start]
  lc = word_tokenize(lc)

  start = len(lc)
  end = start + len(tok_answer)

  row['start'] = start
  row['end'] = end

  return row


def tokenize(dataset, tokenizer, char_tokenizer):

  """
  tokenize the given dataset
  """

  _dataset = dataset.copy()

  tokenized_questions = _dataset['question'].apply(word_tokenize).to_list()
  tokenized_contexts = _dataset['context'].apply(word_tokenize).to_list()

  t_q = tokenizer.texts_to_sequences(tokenized_questions)
  t_c = tokenizer.texts_to_sequences(tokenized_contexts)

  c_q = []
  c_c = []

  for question, context in zip(tokenized_questions, tokenized_contexts):
    _q = char_tokenizer.texts_to_sequences(question)
    _c = char_tokenizer.texts_to_sequences(context)
    c_q.append(_q)
    c_c.append(_c)

  _dataset['tokenized_question'] = t_q
  _dataset['tokenized_context'] = t_c

  _dataset['char_tokenized_question'] = c_q
  _dataset['char_tokenized_context'] = c_c

  return _dataset

def split(dataset, test_size = 0.2, random_state = 42):

  """
  split the dataset in two part: the training and the validation
  """

  # random_state for deterministic state
  tr, vl = train_test_split(dataset, test_size = test_size, random_state = random_state)
  tr.reset_index(drop = True, inplace = True)
  vl.reset_index(drop = True, inplace = True)

  return tr,vl


def df_to_json(df, path, with_answer = True):

  """
  parse the given dataframe into the SQUAD json format and
  save it
  """
  
  data = []

  for title, articles in df.groupby('title'):
    chapter = {'title': title}
    paragraphs = []
    for context, contents in articles.groupby('context'):
      paragraph = {'context': context}
      qas = []
      for i, content in contents.iterrows():
        if with_answer:
          qa = {'answers': [{'answer_start': content['answer_start'], 'text': content['text']}], 'question': content['question'], 'id': content['id']}
        else:
          qa = {'question': content['question'], 'id': content['id']}
        qas.append(qa)
      paragraph.update({'qas': qas})
      paragraphs.append(paragraph)
    chapter.update({'paragraphs': paragraphs})
    data.append(chapter)
  raw_data = {'data': data}

  with open(path, 'w') as handle:
    json.dump(raw_data, handle)

  print(f'dataset saved in {path}')