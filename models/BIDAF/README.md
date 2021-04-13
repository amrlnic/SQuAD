# BIDAF model

* run the bidaf_preprocessing notebook
* run the bidaf_model notebook

you can now make predictions by using the script `compute_answer.py`

### Command

`python compute_answer.py FILE [arguments]`

### Arguments

| Argument | Default | Description |
|:---------|:--------|:------------|
| `FILE` | REQUIRED | The test file |
| `--question_maxlen` | `25` | Maximum length imposed on questions |
| `--context_maxlen` | `400` | Maximum length imposed on contexts |
| `--word_maxlen` | `15` | Maximum length imposed on each word |
| `--batch_size` | `10` | The batch size |
| `--word_tokenizer` | `utils/tokenizers/word_tokenizer.pkl` | Path to the word_tokenizer |
| `--char_tokenizer` | `utils/tokenizers/char_tokenizer.pkl` | Path to the char_tokenizer |
| `--output_file` | `predictions.json` | Path to the output file |
| `--weights` | `utils/models/weights/bidaf_weights` | Path to the weights |
| `--embedding_size` | `300` | The embedding size |
| `--embedding_matrix` | `utils/data/embedding.npy` | Path to the embedding matrix npy file |
| `--learning_rate` | `5e-4` | The learning rate |
| `--filter_size` | `3` | The filter size |
| `--char_embedding_size` | `8` | Print statistics for the first solution |
| `--epochs` | `10` | The number of epochs |


> You only need to run both notebooks for preprocessing / training / inference.
> The utils folder is where files (preprocessed datasets, tokenizers...) are saved.
> You can also find the same functions, classes as in the notebooks (they are imported in the `compute_answers.py` script)

### Default

`python compute_answer.py TEST_SET_PATH`

# Results

> training on the SQUAD V1.1 dataset  
> F1-score: 65.81 %
> Exact Match: 51.08%
