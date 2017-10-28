# StockMarketML (Not Done)

Using the magic of machine learning to predict the trends of the stock market.

## How

### Tools/libs

* [Keras](https://keras.io/)
* [NLTK](http://www.nltk.org/)
* [Gensim](https://radimrehurek.com/gensim/)
* [Numpy](http://www.numpy.org/)
* [Praw](https://praw.readthedocs.io/en/latest/)
* [Requests](http://docs.python-requests.org/en/master/)

### Code

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
