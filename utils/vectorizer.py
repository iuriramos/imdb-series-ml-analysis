from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline

def get_vectorizer(use_hashing, use_tf_idf, use_stop_words, ngram_range, n_features=5000):
    stop_words = 'english' if use_stop_words else None

    if use_hashing:
        if use_tf_idf:
            return make_pipeline(
                            HashingVectorizer(
                                n_features=n_features,
                                stop_words=stop_words,
                                ngram_range=ngram_range),
                            TfidfTransformer())
        else:
            return HashingVectorizer(
                        n_features=n_features,
                        stop_words=stop_words,
                        ngram_range=ngram_range)
    else:
        if use_tf_idf:
            return make_pipeline(
                            CountVectorizer(
                                stop_words=stop_words,
                                ngram_range=ngram_range),
                            TfidfTransformer())
        else:
            return  CountVectorizer(
                            stop_words=stop_words,
                            ngram_range=ngram_range)

if __name__ == '__main__':
    pass
