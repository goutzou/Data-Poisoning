# Data-Poisoning

## Steps:

### 1- Transform our IMDb review sentences into RoBERTa embeddings using thei rRoBERTa embedding API
### 2- Train a logisitic regression model using our RoBERTa embeddings dataset
### 3- Randomly sample test datapoints from test set
### 3- Compute the influence of each training datapoint on each of our sampled test inputs
### 4- Select top-k influencial training datapoints for each test input sampled
### 5- Apply the data poisoning technique on those k influencing training datapoints that influenced our test inputs' clasification
### 5- Transform our poisoned IMDb review sentences into RoBERTa embeddings using thei rRoBERTa embedding API
### 6- Retrain our logisitic regression model on our poisoned RoBERTa embeddings dataset
### 7- Recompute the influence of each training (poisoned) datapoint on each of our sampled test inputs (the exact same one we used previously)
### 8- Analyze and compare how poisoning the dataset changed the classification results of our logistic regression model


