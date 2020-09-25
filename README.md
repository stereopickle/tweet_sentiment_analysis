# Tweet Sentiment Analysis

## Overview
The goal of the project is to develop a classification model that classifies whether a tweet about a product or brand is positive, negative or neutral. 

## Repo Structure

## Data 
Data was sourced from [data.world](https://data.world/crowdflower/brands-and-product-emotions), which provided over 8000 tweets that have been classified to be whether positive, negative, or neutral by human contributors. 

[!target distribution](/PNG/target_distribution.png)
After removing all tweets classified to be "I can't tell", the tweets were split into train, valid, test sets (7:1.5:1.5). Our final training data consists of 2,651 tweets with no emotion, 1,545 positive, and 292 negative. 

## NLP Preprocessing
A standard NLP preprocessing was applied. Details are as below...
1. Converted to lowercase
2. Removed HTML symbols, punctuations, digits, links, mentions
3. Any words that can bias the results removed (e.g. brand, product, event names)
4. Lemmatized using WordNetLemmatizer
5. Tokenized

In order to reduce overall dimensions, similarities between low frequency words and other words were computed and low-frequency words were replaced with the most similar word if their cosine similarity was above .8. 

Additionally, the percentage of capitalized characters, a number of exclamation points, question marks, links, mentions, tags were computed for additional analysis. 


## Evaluation
### Evaluation Metrics
Both sensitivity and precision are equally important in our business case, so we used f1-score as our evaluation metrics. We took the macro-average across all classes, as we have pretty significant class imbalance.

### Modeling
Largely I used the Bag of Words approach and the deep learning approach. 
For the Bag of Words models, I calculated the simple count vectors and TF-IDF vectors and ran it through the Naive Bayes and SVM. For deep learning approach, an LSTM model was tested with and without the GloVe word embedding weights. Lastly BERT 


## Results


