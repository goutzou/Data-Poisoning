# Data-Poisoning

## Resources:

### Dataset: https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews

### Embeddings: https://huggingface.co/docs/transformers/en/model_doc/roberta

### Influence functions paper: https://arxiv.org/abs/1703.04730

## Steps:

### 1- Transform our IMDb review sentences into RoBERTa embeddings using thei RoBERTa embedding API

### Files- generate_roberta_embeddings.py

### 2- Train a logisitic regression model using our RoBERTa embeddings dataset

### Files- train_logistic_regression.py, train_test_split.npz

### 3- Randomly sample test datapoints from test set

### 4- Compute the influence of each training datapoint on each of our sampled test inputs

### Files- compute_influence2.py, influence_scores_chunk_0 to 9.csv. (The difference between compute_influence and compute_influence2 is minimal, primarily paths and small stuff. I just kept both because I ran compute_influence2 for the experiment, and we had originally come up with the regular file. Don't let it confuse you.)

### Command To Run compute_influence2.py: python3 compute_influence.py --chunk_index X --total_chunks 10, where X the chunk from 0 to 9

### 5- Select top-k influencial training datapoints for each test input sampled

### Files- k_most.py, influence_scores_merged_top_1percent_abs.csv

### 6- Apply the data poisoning technique on those k influencing training datapoints that influenced our test inputs' clasification

### Files- binary_imdb_data_with_ids_poisoned.csv, flipping_labels.py

### 7- Transform our poisoned IMDb review sentences into RoBERTa embeddings using thei rRoBERTa embedding API

### Files- roberta_embeddings_poisoned.csv, roberta_embeddings_poisoned.pt, generate_roberta_embeddings2.py

### 8- Retrain our logisitic regression model on our poisoned RoBERTa embeddings dataset

### Files- train_poisoned.py, train_test_split_poison.npz (And here the performance metrics of the model not only did not drop, but increased.)

### 9- Recompute the influence of each training (poisoned) datapoint on each of our sampled test inputs (the exact same one we used previously)

### 10- Analyze and compare how poisoning the dataset changed the classification results of our logistic regression model
