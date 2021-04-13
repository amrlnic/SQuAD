import argparse
import os
from utils.datasets import SQUAD_dataset
from utils.models import BIDAF
from utils import load_dataset, clean_dataset, tokenize
import pickle
import numpy as np


def main():
	parser = argparse.ArgumentParser(description = 'BIDAF')
	parser.add_argument('file', type = str, help = 'the test file')
	parser.add_argument('--question_maxlen', default = 25, type = int)
	parser.add_argument('--context_maxlen', default = 400, type = int)
	parser.add_argument('--word_maxlen', default = 15, type = int)
	parser.add_argument('--batch_size', default = 10, type = int)
	parser.add_argument('--word_tokenizer', default = 'utils/tokenizers/word_tokenizer.pkl', type = str, help = 'path to the word_tokenizer')
	parser.add_argument('--char_tokenizer', default = 'utils/tokenizers/char_tokenizer.pkl', type = str, help = 'path to the char_tokenizer')
	parser.add_argument('--output_file', default = 'predictions.json', type = str, help = 'path to the output file')
	parser.add_argument('--weights', default = 'utils/models/weights/bidaf_weights', type = str, help = 'path to the weights')
	parser.add_argument('--embedding_size', default = 300, type = int)
	parser.add_argument('--embedding_matrix', default = 'utils/data/embedding.npy', type = str, help = 'path to the embedding matrix npy file')
	parser.add_argument('--learning_rate', default = 0.0005, type = float)
	parser.add_argument('--filter_size', default = 3, type = int)
	parser.add_argument('--char_embedding_size', default = 8, type = int)
	parser.add_argument('--epochs', default = 10, type = int)
	args = parser.parse_args()

	QUESTION_MAXLEN = args.question_maxlen
	CONTEXT_MAXLEN = args.context_maxlen
	WORD_MAXLEN = args.word_maxlen
	BATCH_SIZE = args.batch_size
	LR = args.learning_rate
	EMBEDDING_SIZE = args.embedding_size
	N_FILTERS = EMBEDDING_SIZE
	CHAR_EMBEDDING_SIZE = args.char_embedding_size
	EPOCHS = args.epochs
	FILTER_SIZE = args.filter_size

	curr = os.getcwd()
	
	filepath = os.path.join(curr, args.file)
	output_path = os.path.join(curr, args.output_file)
	word_tokenizer_path = os.path.join(curr, args.word_tokenizer)
	char_tokenizer_path = os.path.join(curr, args.char_tokenizer)
	weights_path = os.path.join(curr, args.weights)
	embedding_matrix_path = os.path.join(curr, args.embedding_matrix)

	with open(word_tokenizer_path, 'rb') as word_handle:
		word_tokenizer = pickle.load(word_handle)

	with open(char_tokenizer_path, 'rb') as char_handle:
		char_tokenizer = pickle.load(char_handle)

	embedding_matrix = np.load(embedding_matrix_path)

	WORD_VOCAB_LEN = len(word_tokenizer.word_index) + 1
	CHAR_VOCAB_LEN = char_tokenizer.num_words

	dataset = load_dataset(filepath, with_answer = False)
	SAMPLES = dataset.shape[0]

	print('[INFO] cleaning data...')
	dataset = clean_dataset(dataset, with_answer = False)
	print('[INFO] done !')
	print('[INFO] tokenizing data...')
	dataset = tokenize(dataset, word_tokenizer, char_tokenizer)
	print('[INFO] done !')
	dataset = dataset[(dataset['tokenized_question'].str.len() <= QUESTION_MAXLEN) & (dataset['tokenized_context'].str.len() <= CONTEXT_MAXLEN)].reset_index(drop = True) 
	print(f'[PREPROCESSING] we get rid of : {SAMPLES - dataset.shape[0]} samples')

	dataset = SQUAD_dataset(dataset, 
		batch_size = BATCH_SIZE, 
		question_maxlen = QUESTION_MAXLEN,
		context_maxlen = CONTEXT_MAXLEN, 
		word_maxlen = WORD_MAXLEN, 
		with_answer = False)

	bidaf_model = BIDAF(
		QUESTION_MAXLEN,
		CONTEXT_MAXLEN,
		WORD_VOCAB_LEN,
		EMBEDDING_SIZE,
		embedding_matrix,
		CHAR_VOCAB_LEN,
		WORD_MAXLEN,
		N_FILTERS,
		FILTER_SIZE,
		CHAR_EMBEDDING_SIZE,
		word_tokenizer_path,
		char_tokenizer_path)

	bidaf_model.load_weights(weights_path)
	print('[INFO] making predictions...')
	bidaf_model.multi_predictions([dataset], output_path)

if __name__ == '__main__':
	main()