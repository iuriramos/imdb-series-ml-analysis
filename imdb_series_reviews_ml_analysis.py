import os
import re
import sys
import time
import argparse

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, StratifiedKFold

from utils.vectorizer import get_vectorizer
from utils.model import model
from utils.fetch_imdb_crime_series_reviews import fetch_imdb_crime_series_reviews

class imdb_series_reviews_ml_analysis:
    def __init__(self, top=25, neg_rating_range=None, pos_rating_range=None):
        self.top = top
        self.neg_rating_range = neg_rating_range or (1, 2, 3)
        self.pos_rating_range = pos_rating_range or (8, 9, 10)

    def report(self, vectorizer_params, model_name, show_results):
        # Perform grid search on model_params
        def grid_search(vect, model_, model_params):
            # steps to make pipeline
            pipeline = Pipeline([('vect', vect), ('model', model_)])

            wrapper = dict(model=model_params)
            # params in GridSearchCV format
            grid_params = {step_name + '__' + param_name: param_values
                                    for step_name, step_params in wrapper.items()
                                    for param_name, param_values in step_params.items()}

            # Stratified K-Fold for cross validation
            kfold = StratifiedKFold(y_train, n_folds=10, random_state=42)
            grid = GridSearchCV(pipeline, param_grid=grid_params, cv=2, n_jobs=4, verbose=1, error_score=0)
            grid.fit(X_train, y_train)

            # Present model results
            print '*' * 20
            best_model = grid.best_estimator_.steps[-1]
            print 'estimator = ', best_model
            print 'score (train) = ', grid.best_score_
            print 'score (test) = ', grid.score(X_test, y_test)

        X, y = fetch_imdb_crime_series_reviews(self.top,
            self.neg_rating_range, self.pos_rating_range)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify = y, random_state=42)

        # vectorizer
        vect = get_vectorizer(**vectorizer_params)
        # perform grid_search on params
        if show_results:
            model_, params = model.get_model_params(model_name)
            grid_search(vect, model_, params)
        else:
            model_ = model.get_model(model_name)
            grid_search(vect, model_, params={})

def run():
    # parse command line params
    parser = argparse.ArgumentParser(description='ML Classifier for imdb crime series reviews')
    parser.add_argument('-t', '--top', type=int, default=25, dest='top', help='Top crime series to download')
    parser.add_argument('-neg', '--max-negative-rating', type=int, default=3, choices=range(1, 6), dest='max_neg_rating', help='Maximum rating for a review to be considered negative')
    parser.add_argument('-pos', '--min-pos-rating', type=int, default=8, choices=range(6, 11), dest='min_pos_rating', help='Minimum rating for a review to be considered positive')

    parser.add_argument('-n', '--n-features', type=int, default=10000, dest='n_features', help='Maximum number of features to extract from text (only used in HashingVectorizer)')
    parser.add_argument('--no-stop-words', action='store_true', default=False, dest='no_stop_words', help='Do not erase stop-words.')
    parser.add_argument('-ngram-range', default='(1, 1)', choices=('(1, 1)', '(1, 2)', '(1, 3)'), dest='ngram_range', help='Ngram range used for text parsing. [choices available: (1, 1) (default), (1,2), (1, 3)')
    parser.add_argument('--no-hashing', action='store_true', default=False, dest='no_hashing', help='Do not use hashing in feature extraction')
    parser.add_argument('--no-tf-idf', action='store_true', default=False, dest='no_tf_idf', help='Do not use TF-IDF feature extraction')
    parser.add_argument('-m', '--model', default='RandomForestClassifier', choices=('RandomForestClassifier', 'LogisticRegression', 'LinearSVC', 'SVC', 'SGDClassifier'), dest='model_name', help='ML model to classify reviews')
    parser.add_argument('-r', '--show-results', action='store_true', default=False, dest='show_results', help='Show results for all combinations of parameters')

    args = parser.parse_args()

    imdb = imdb_series_reviews_ml_analysis(top=args.top, neg_rating_range=range(1, args.max_neg_rating + 1), pos_rating_range=range(args.min_pos_rating, 11))
    vectorizer_params = {
        'use_hashing': not args.no_hashing,
        'use_tf_idf': not args.no_tf_idf,
        'use_stop_words': not args.no_stop_words,
        'ngram_range': eval(args.ngram_range),
        'n_features': args.n_features,
    }
    imdb.report(vectorizer_params=vectorizer_params, model_name=args.model_name, show_results=args.show_results)

def run_all_settings():
    imdb = imdb_series_reviews_ml_analysis()
    dir_path = './results'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    prefix = []
    # for use_hashing in False, True:
    for use_hashing in True, :
        prefix.append('hash_10k' if use_hashing else 'count')
        for use_tf_idf in False, True:
            prefix.append('tf_idf' if use_tf_idf else '')
            for use_stop_words in False, True:
                prefix.append('sw' if use_stop_words else '')
                for ngram_range in ((1, 1), (1, 2), (1, 3)):
                    prefix.append('ngram_' + str(ngram_range))
                    vectorizer_params = {
                        'use_hashing': use_hashing,
                        'use_tf_idf': use_tf_idf,
                        'use_stop_words': use_stop_words,
                        'ngram_range': ngram_range}
                    for model_name in ('RandomForestClassifier', 'LogisticRegression',
                                                    'LinearSVC', 'SVC', 'SGDClassifier'):
                        prefix.append(model_name)
                        file_path = os.path.join(dir_path, '_'.join(prefix) + '.txt')
                        with open(file_path, 'w') as f:
                            sys.stdout = f
                            imdb.report(
                                        vectorizer_params=vectorizer_params,
                                        model_name=model_name,
                                        show_results=True)
                        prefix.pop()
                    prefix.pop()
                prefix.pop()
            prefix.pop()
        prefix.pop()
if __name__ == '__main__':
    # run()
    run_all_settings()
