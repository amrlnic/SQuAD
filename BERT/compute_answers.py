import argparse
import os
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from util import load_dataset, clean_dataset, skip, create_inputs_targets, create_model, predict
import pickle
import numpy as np
import json
from utils.model import BERT

tf.get_logger().setLevel('FATAL')

def main():
    parser = argparse.ArgumentParser(description='BERT')
    parser.add_argument('file', type=str, help='the test file')
    parser.add_argument('--text_maxlen', default=384, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--output_file', default='predictions.json',
                        type=str, help='path to the output file')
    parser.add_argument('--weights', default='BERT\\utils\\model\\weights\\bert_bilstm_noDrop_weights.h5',
                        type=str, help='path to the weights')
    parser.add_argument('--enc_dec', default="True", type=str)
    parser.add_argument('--enc_dim', default=128, type=int)
    parser.add_argument('--dec_dim', default=64, type=int)
    parser.add_argument('--rec_mod', default='biLSTM', type=str)
    parser.add_argument('--bert_ft', default="True", type=str)
    parser.add_argument('--dropout', default="False", type=str)
    parser.add_argument('--drop_prob', default=0.5, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    args = parser.parse_args()

    TEXT_MAXLEN = args.text_maxlen
    BATCH_SIZE = args.batch_size
    ENC_DEC = bool(args.enc_dec)
    ENC_DIM = args.enc_dim
    DEC_DIM = args.dec_dim
    REC_MOD = args.rec_mod
    BERT_FT = bool(args.bert_ft)
    DROPOUT = bool(args.dropout)
    DROP_PROB = args.drop_prob
    EPOCHS = args.epochs

    curr = os.getcwd()

    filepath = os.path.join(curr, args.file)
    output_path = os.path.join(curr, args.output_file)
    weights_path = os.path.join(curr, args.weights)

    # Save the slow pretrained tokenizer
    slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    save_path = "bert_base_uncased/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

    # Load the fast tokenizer from saved file
    tokenizer = BertWordPieceTokenizer(
        "bert_base_uncased/vocab.txt", lowercase=True)

    with open(filepath) as file:
        data = json.load(file)

    # print("###############"+str(data)+"####################")
    # print("###############"+str(type(data))+"####################")
    dataset = load_dataset(data)
    SAMPLES = dataset.shape[0]

    print('[INFO] cleaning data...')
    dataset = clean_dataset(dataset)

    # takes a while
    dataset = dataset.apply(skip, tokenizer=tokenizer,
                            text_max_len=TEXT_MAXLEN, axis=1)
    # we get rid of samples where the answer doesn't match the context
    dataset = dataset[dataset['skip'] == False]
    print(dataset)

    print('[INFO] done !')

    print('[INFO] creating input targets...')
    x, y = create_inputs_targets(dataset)
    print('[INFO] done !')

    """bert_model = BERT(
        ENC_DEC,
        ENC_DIM,
        DEC_DIM,
        REC_MOD,
        BERT_FT,
        DROPOUT,
        DROP_PROB,
        TEXT_MAXLEN)"""

    bert_model = create_model(ENC_DEC,
                              ENC_DIM,
                              DEC_DIM,
                              REC_MOD,
                              BERT_FT,
                              DROPOUT,
                              DROP_PROB,
                              TEXT_MAXLEN)

    bert_model.load_weights(weights_path)
    print('[INFO] making predictions...')
    predict(bert_model, x[:5], output_path, tokenizer)


if __name__ == '__main__':
    main()
