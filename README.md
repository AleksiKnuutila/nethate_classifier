# Online Hate research project - slur identification

This repository is related to the research project "Online Hate - The networks, practices and motives of the producers and distributors of hate speech". The project page, including the published results, can be found [here](https://vnk.fi/-/1927382/vihan-verkot-vihapuheen-tuottajien-ja-levittajien-motiivit-verkostot-ja-toimintamuodot).

The repository contains some of the resources used to identify slurs targeted towards minorities on digital media. These are published for purposes of transparency and to make it possible to evaluate the research results.

# Training data

annotated_tweet_ids.csv contains IDs of tweets and their annotations, following [Twitter's policy on data publication](https://developer.twitter.com/en/developer-terms/more-on-restricted-use-cases). The tweet metadata needs to be obtained from their API to train the model. Our training data also contained annotated messages from other platforms, which we could not publish.

# Classifier

We used a relatively simple model transformers-based classifier with a pretrained model for the identification task.

The model can be run in the following way:

```
pip install transformers datasets pandas numpy
python train.py
``` 
