import numpy as np
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier

class model(object):
    @staticmethod
    def get_model(model_name):
        if model_name == 'RandomForestClassifier':
            return RandomForestClassifier(random_state=42, n_jobs=4)
        elif model_name == 'MultinomialNB':
            return MultinomialNB()
        elif model_name == 'BernoulliNB':
            return BernoulliNB()
        elif model_name == 'SVC':
            return SVC(kernel='rbf', random_state=42)
        elif model_name == 'NuSVC':
            return NuSVC(random_state=42)
        elif model_name == 'LinearSVC':
            return LinearSVC(random_state=42)
        elif model_name == 'LogisticRegression':
            return LogisticRegression(random_state=42, n_jobs=4)
        # elif model_name == 'SGDClassifier':
        #     return SGDClassifier(random_state=42, n_jobs=4)
        else:
            raise ValueError("{} not available".format(model_name))

    @staticmethod
    def get_model_params(model_name):
        model = get_model(model_name)
        if model_name == 'RandomForestClassifier':
            return model, dict(n_estimators=(1, 10, 100), min_samples_split=(2, 5, 10))
        elif model_name == 'MultinomialNB':
            return model, dict(alpha=np.linspace(0.1, 1, 10))
        elif model_name == 'BernoulliNB':
            return model, dict(alpha=np.linspace(0.1, 1, 10))
        elif model_name == 'SVC':
            return model, dict(C=np.logspace(-1, 3, 5), gamma=np.logspace(-4, 0, 5))
        elif model_name == 'NuSVC':
            return model, dict(nu=np.linspace(0.1, 1, 10), gamma=np.logspace(-4, 0, 5))
        elif model_name == 'LinearSVC':
            return model, dict(C=np.logspace(-1, 3, 5))
        elif model_name == 'LogisticRegression':
            return model, dict(C=np.logspace(-1, 3, 5))
        # elif model_name == 'SGDClassifier':
        #    return model, dict()

if __name__ == '__main__':
    pass
