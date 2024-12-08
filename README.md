# Data-Poisoning

## Resources: 
### Dataset: [!Link] https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews
### Embeddings: [!Link] https://huggingface.co/docs/transformers/en/model_doc/roberta
### Influence functions paper: [!Link] https://arxiv.org/abs/1703.04730

## Steps:

### 1- Transform our IMDb review sentences into RoBERTa embeddings using thei RoBERTa embedding API
### 2- Train a logisitic regression model using our RoBERTa embeddings dataset
### 3- Randomly sample test datapoints from test set
### 4- Compute the influence of each training datapoint on each of our sampled test inputs
### 5- Select top-k influencial training datapoints for each test input sampled
### 6- Apply the data poisoning technique on those k influencing training datapoints that influenced our test inputs' clasification
### 7- Transform our poisoned IMDb review sentences into RoBERTa embeddings using thei rRoBERTa embedding API
### 8- Retrain our logisitic regression model on our poisoned RoBERTa embeddings dataset
### 9- Recompute the influence of each training (poisoned) datapoint on each of our sampled test inputs (the exact same one we used previously)
### 10- Analyze and compare how poisoning the dataset changed the classification results of our logistic regression model


