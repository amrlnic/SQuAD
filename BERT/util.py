import json
import pandas as pd
import numpy as np

def load_dataset(file, record_path = ['data', 'paragraphs', 'qas', 'answers'], verbose = True):

  """
  parse the SQUAD dataset into a dataframe
  """

  if verbose:
      print("Reading the json file")

  if verbose:
      print("[INFO] processing...")

  # parsing different level's in the json file
  js = pd.json_normalize(file , record_path )
  m = pd.json_normalize(file, record_path[:-1] )
  r = pd.json_normalize(file, record_path[:-2])
  t = pd.json_normalize(file, record_path[0])

  title = pd.json_normalize(file['data'], record_path = ['paragraphs'], meta = 'title')

  #combining it into single dataframe
  idx = np.repeat(r['context'].values, r.qas.str.len())
  ndx  = np.repeat(m['id'].values, m['answers'].str.len())
  m['context'] = idx
  m['title'] = np.repeat(title['title'].values, r.qas.str.len())
  js['q_idx'] = ndx
  main = pd.concat([ m[['id','question','context', 'title']].set_index('id'), js.set_index('q_idx')], 1, sort = False).reset_index()
  main['c_id'] = main['context'].factorize()[0]
  if verbose:
      print(f"[INFO] there are {main.shape[0]} questions with single answer")
      print(f"[INFO] there are {main.groupby('c_id').sum().shape[0]} different contexts")
      print(f"[INFO] there are {len(t)} unrelated subjects")
      print("[INFO] Done")
  return main

SAMPLES = squad_dataset.shape[0]

def preprocess_sentence(text):

  """
  lowercase and strip the given text
  """

  text = text.lower()
  text = text.strip()
  return text

def clean_dataset(dataset):

  """
  preprocess the dataset
  """

  _dataset = dataset.copy()

  cleaned_questions = _dataset['question'].apply(preprocess_sentence)
  cleaned_texts = _dataset['text'].apply(preprocess_sentence)

  # we process only different contexts and then we duplicate them
  unique_context = pd.Series(_dataset['context'].unique())
  count_c = _dataset.groupby('c_id').count()['text']
  cleaned_contexts = unique_context.apply(preprocess_sentence)

  _dataset['question'] = cleaned_questions
  _dataset['text'] = cleaned_texts
  _dataset['context'] = pd.Series(np.repeat(cleaned_contexts, count_c).tolist())

  return _dataset

def skip(row):

  """
  find the rows for which an answer can't be found
  """


  answer = row['text']
  context = row['context']
  start_char_idx = row['answer_start']
  question = row['question']

  # initialize skip column
  row['skip'] = False


  # Find end character index of answer in context
  end_char_idx = start_char_idx + len(answer)
  if end_char_idx >= len(context):
    row['skip'] = True
    return row

  # Mark the character indexes in context that are in answer
  is_char_in_ans = [0] * len(context)
  for idx in range(start_char_idx, end_char_idx):
    is_char_in_ans[idx] = 1

  # Tokenize context
  tokenized_context = tokenizer.encode(context)
  #row['tokenized context'] = tokenized_context

  # Find tokens that were created from answer characters
  ans_token_idx = []
  for idx, (start, end) in enumerate(tokenized_context.offsets):
    if sum(is_char_in_ans[start:end]) > 0:
      ans_token_idx.append(idx)

  if len(ans_token_idx) == 0:
    row['skip'] = True
    return row

  # Tokenize question
  tokenized_question = tokenizer.encode(question)
  #row['tokenized question'] = tokenized_question

  # Inputs of the model: here are used to determine whether to skip the row or not
  input_ids = tokenized_context.ids + tokenized_question.ids[1:]
  token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )


  padding_length = MAX_LEN - len(input_ids)

  if padding_length < 0:
    row['skip'] = True

  return row


def split(dataset, test_size = 0.2, random_state = 42):

  """
  split the dataset in two part: the training and the validation
  """

  # random_state for deterministic state
  tr, vl = train_test_split(dataset, test_size = test_size, random_state = random_state)
  tr.reset_index(drop = True, inplace = True)
  vl.reset_index(drop = True, inplace = True)

  return tr,vl

def df_to_json(df, path):

  """
  parse the given dataframe into the SQUAD json format
  """
  
  data = []

  for title, articles in df.groupby('title'):
    chapter = {'title': title}
    paragraphs = []
    for context, contents in articles.groupby('context'):
      paragraph = {'context': context}
      qas = []
      for i, content in contents.iterrows():
        qa = {'answers': [{'answer_start': content['answer_start'], 'text': content['text']}], 'question': content['question'], 'id': content['index']}
        qas.append(qa)
      paragraph.update({'qas': qas})
      paragraphs.append(paragraph)
    chapter.update({'paragraphs': paragraphs})
    data.append(chapter)
  raw_data = {'data': data}

  with open(path, 'w') as handle:
    json.dump(raw_data, handle)

  print(f'dataset saved in {path}')