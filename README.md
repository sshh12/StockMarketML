# StockMarketML (WIP)

Using the magic of machine learning to predict the trends of the stock market.

## App

Applying the Model

#### Tools/libs Used
* [Keras](https://keras.io/)
* [Numpy](http://www.numpy.org/)
* [Praw](https://praw.readthedocs.io/en/latest/)
* [Requests](http://docs.python-requests.org/en/master/)

```#TODO```

## Lab 2

Current (Second) Attempt

##### CollectData

This script gathers data by scraping websites and does basic word processing.

##### HeadlineEmbeddings

This uses a model to extract the features from headlines.

##### TrendPrediction

This script takes window of stock prices and predicts next close price.

## Lab 1

First Attempt

##### CollectData

This script gathers data by scraping websites and does basic word processing.

##### LoadData

This helper script loads the csv files and preprocesses data before being used in a model.

##### BasicPredictionClassification

This uses a window of the last n stock closes and volumes to predict whether the next close with be high or lower than it opened.

##### BasicPredictionRegression

This uses a window of the last n stock prices to predict the next close price.

##### HeadlinePredictionClassification

This uses headlines processed through doc2vec to predict changes in close price.

##### HeadlineAndTickerClassification

Using historic stock prices and headlines to predict close price.
