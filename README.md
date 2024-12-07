# Data-Poisoning

## Steps:

### 1- Train RoBERTa on our dataset with our manual selection of the loss function
### 2- Get a mathematical formula for the loss function that we can use in code
### 3- Get the k influencing datapoints that influenced the classification on a random sample of test inputs
### 4- Apply the data poisoning technique on our randomly sampled test inputs
### 5- Train RoBERTa again on our dataset that includes the poisoned data
### 6- Repeat steps 2-3
### 7- Compare and analyze results from step 3 and 6
