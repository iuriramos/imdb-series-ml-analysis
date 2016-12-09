# Machine Learning applied in IMDB reviews of crime series

A great source of information for sentiment analysis is review websites, such as Amazon, TripAdvisor, Glassdoor, etc. In this project, I will analyze The Internet Movie Database (IMDB), specifically IMDB reviews for top crime series.

## Motivation

Sentiment Analysis is a major area in Natural Language Processing and consists of extracting underlying sentiments from texts. Its ultimate goal is to determine whether the sentiment related to the text is positive, negative or neutral.

The ultimate goal of this project is to use supervised models to classify IMDB reviews of top crime series in two groups that express positive or negative sentiments.

## Installation

All dependencies for this project can be found in the _requirements.txt_ file.

## Usage

The data is downloaded from [IMDB Top Crime Series](http://www.imdb.com/search/title?genres=crime&title_type=tv_series,mini_series). A structure of directories will be set up in the project's directory. Using `imdb_series_reviews_ml_analysis.py` script, a number of parameters can be set up for the download:
* Use `--top` to define how many series to download.
* `-neg` defines the maximum rating star associated with a negative sentiment. The minimum rating star associated with a negative sentiment is constant and equals to 1. So, rating stars in the interval `[1, neg]` are associated with negative sentiments.
* `-pos` defines the minimum rating star associated with a positive sentiment. The maximum rating star associated with a positive sentiment is constant and equals to 10. So, rating stars in the interval `[pos, 10]` are associated with positive sentiments.

This project uses the bag of words approach to transform text reviews in feature vectors. We can manipulate this structure by using some arguments. The bag of words structure set up uses feature hashing and TF-IDF, by default. It also removes stop words from the text and uses only words as tokens (`ngram_range` set to `(1, 1)`).

However, we can remove feature hashing and TF-IDF from analysis by typing  `--no-hashing`, and `--no-tf-idf`, respectively. You can change how to deal with stop words and ngrams using `--no-stop-words` and `--ngram-range` arguments. We can also change the default number of features in feature hashing using the argument `-n` followed by the number of features (5000, 10000, etc).

The feature vectors go to fit the supervised learning models. To select the model, use `-m` followed by the name of the model in the list: [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [BernoulliNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB), [MultinomialNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html), [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC), [NuSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC), and [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) from Scikit Learn.

If you want to perform a grid search in the selected model, use the flag `-r` in the command line. A search for the best combination of parameters will be made in this case. You can also change the parameters and values that will be passed to the grid search by changing the `utils/model.py` file.

For a better understanding of these command line arguments, type `python imdb_series_reviews_ml_analysis.py --help`.

## Example

```
python imdb_series_reviews_ml_analysis.py --help
python imdb_series_reviews_ml_analysis.py
python imdb_series_reviews_ml_analysis.py -t 25 -neg 3 -pos 8
python imdb_series_reviews_ml_analysis.py -n 10000
python imdb_series_reviews_ml_analysis.py --no-stop-words
python imdb_series_reviews_ml_analysis.py --ngrams-range (1, 2)
python imdb_series_reviews_ml_analysis.py --no-hashing
python imdb_series_reviews_ml_analysis.py -m LogisticRegression
python imdb_series_reviews_ml_analysis.py -m LogisticRegression --no-hashing --no-stop-words --ngrams-range (1, 2) -r
```

You can find more information in *capstone_project.pdf*.
