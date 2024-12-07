# Data-Poisoning

## Steps:

### 1- Transform our IMDb review sentences into RoBERTa embeddings using thei RoBERTa embedding API
### 2- Train a logisitic regression model using our RoBERTa embeddings dataset
### 4- Randomly sample test datapoints from test set
### 5- Compute the influence of each training datapoint on each of our sampled test inputs
### 6- Select top-k influencial training datapoints for each test input sampled
### 7- Apply the data poisoning technique on those k influencing training datapoints that influenced our test inputs' clasification
### 8- Transform our poisoned IMDb review sentences into RoBERTa embeddings using thei rRoBERTa embedding API
### 9- Retrain our logisitic regression model on our poisoned RoBERTa embeddings dataset
### 10- Recompute the influence of each training (poisoned) datapoint on each of our sampled test inputs (the exact same one we used previously)
### 11- Analyze and compare how poisoning the dataset changed the classification results of our logistic regression model


