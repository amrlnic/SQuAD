# BERT model

* run the BERT notebook

you can now make predictions by using the script `compute_answer.py`

### **Command:**

`python compute_answer.py [arguments]`

### **Arguments:**
| Argument | Default | Description |
|:---------|:--------|:------------|
| `FILE` | REQUIRED | The test file |
| `--text_maxlen` | `384` | Maximum length imposed on text |
| `--batch_size` | `256` | The batch size |
| `--output_file` | `predictions.json` | Path to the output file |
| `--weights` | `BERT/utils/model/weights/bert_bilstm_noDrop_weights.h5` | Path to the weights |
| `--enc_dec` | `400` | Use the encoder decoder model or not(If False, the base model will be used), possible: `True`, `False` |
| `--enc_dim` | `128` | The encoding dimension |
| `--dec_dim` | `64` | The decoding dimension |
| `--rec_mod` | `biLSTM` | Set the type of recurrent modules, possible: `biLSTM`, `GRU` |
| `--bert_ft` | `True` | choose to fine-tune the BERT, possible: `True`, `False` |
| `--dropout` | `False` | Use dropout, possible: `True`, `False` |
| `--drop_prob` | `0.5` | The dropout probability |
| `--epochs` | `3` | The number of epochs |

### **Default:**

`python compute_answer.py TEST_SET_PATH`

* You only need to run both notebook for preprocessing / training / inference.
* The utils folder is where files (preprocessed datasets, tokenizers...) are saved.
* You can also find the same functions, classes as in the notebooks (they are imported in the `compute_answers.py` script)


# Results

* BERT Base model (without dropout)
  * F1-score: 73.09 %
  * Exact Match: 58.22 %
* BERT Base model (with dropout)
  * F1-score: 72.78 %
  * Exact Match: 57.58 %
* Encoder decoder model with GRU (without dropout)
  * F1-score: 73.02 %
  * Exact Match: 58.10 %
* Encoder decoder model with GRU (with dropout)
  * F1-score: 72.84 %
  * Exact Match: 57.75 %
* Encoder decoder model with biLSTM (without dropout)
  * F1-score: 73.62 %
  * Exact Match: 58.10 %
* Encoder decoder model with biLSTM (with dropout)
  * F1-score: 73.64 %
  * Exact Match: 58.14 %