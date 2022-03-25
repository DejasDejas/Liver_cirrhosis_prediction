import os.path
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import pickle
import logging
import time
from src.data.make_dataset import load_dataset
from src.data.treatments import Treatment
from config.config import ROOT_DIR

logger: logging.Logger = logging.getLogger(__name__)


def timeit(method):
    """
    python decorator to measure the execution time of methods:
    """

    def timed(*args, **kw):
        time_start = time.time()
        result = method(*args, **kw)
        time_end = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((time_end - time_start) * 1000)
        else:
            logger.info('{} {:.2f} ms'.format(method.__name__, (time_end - time_start) * 1000))
        return result

    return timed


def save_model(_file_name, _model):
    assert '.pkl' in _file_name, "The model name must have .pkl format !"
    load_path = os.path.join(ROOT_DIR, 'models', _file_name)
    pickle.dump(_model, open(load_path, "wb"))
    logger.info('XGBoost model saved with name {}.'.format(_file_name))
    return


def load_model(_file_name):
    assert '.pkl' in _file_name, "The model name must have .pkl format !"
    load_path = os.path.join(ROOT_DIR, 'models', _file_name)
    return pickle.load(open(load_path, "rb"))


def preprocessing_data(dataframe, target_feature, verbose=True):
    """
    split and prepare data for training model
    """
    treatment = Treatment(dataframe, target_feature)
    X_tr, X_te, y_tr, y_te = treatment.split_dataset(dataframe)
    if verbose:
        logger.info('Samples in Train Set:', len(X_tr))
        logger.info('Samples in Test Set:', len(X_te))
    return X_tr, X_te, y_tr, y_te


def training(train_index, test_index, _X, _y, _model):
    """

    @rtype: object
    """
    _y_train = _y[train_index]
    _y_test = _y[test_index]
    _X_train = _X[train_index, :]
    _X_test = _X[test_index, :]
    _model.fit(_X_train, _y_train)
    return _model.score(_X_test, _y_test)


@timeit
def loop_skf_train(_skf, _X, _y, _model):
    acc = []
    for fold_no, (train_index, test_index) in enumerate(_skf.split(_X, _y), start=1):
        score = training(train_index, test_index, _X, _y, _model)
        acc.append(score)
        logger.info('For Fold {} the accuracy is {}'.format(str(fold_no), score))
    logger.info('Logistic Regression Mean Accuracy = ', np.mean(acc))
    return


def xgboost():
    return XGBClassifier(learning_rate=0.75, max_depth=3, random_state=1, gamma=0, eval_metric='error')


def parameter_optimization(_X, _y, _model):
    clf = GridSearchCV(_model,
                       {'max_depth': [2, 3, 4, 5, 6],
                        'n_estimators': [50, 100, 200]}, verbose=1, n_jobs=1)
    clf.fit(_X, _y)
    logger.info(clf.best_score_)
    logger.info(clf.best_params_)
    return clf


def train(_df, _model, _target, _file_name_save):
    X_train, X_test, y_train, y_test = preprocessing_data(_df, _target)
    skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    # parameters optimization:
    clf = parameter_optimization(X_train, y_train, _model)
    _model.set_params(**clf.best_params_)

    # test if model is trained:
    loop_skf_train(skf, X_train, y_train, _model)
    save_model(_file_name_save, _model)
    return


def main():
    df = load_dataset()
    df_nan_mean = load_dataset('Nan_mean_cirrhosis.csv')
    df_outliers = load_dataset('outliers_cirrhosis.csv')
    target = 'Stage'  # target dataset to predict

    model = xgboost()
    model_nan_mean = xgboost()
    model_outliers = xgboost()

    train(df, model, target, 'model.pkl')
    train(df_nan_mean, model_nan_mean, target, 'nan_mean_model.pkl')
    train(df_outliers, model_outliers, target, 'outliers_model.pkl')
    return


if __name__ == '__main__':
    main()
