from sklearn.metrics import classification_report
from src.models.train_model import load_model
import logging
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data.make_dataset import load_dataset
from src.models.train_model import preprocessing_data
from config.config import ROOT_DIR

logger: logging.Logger = logging.getLogger(__name__)


def prediction_shap(_model, _df, _pred, _X_test):
    indexes_predicted = _X_test.index
    _pred = np.array(_pred)
    original = np.array(_df.loc[indexes_predicted, 'Stage'])

    def average_gap(l1, l2):
        result = 0
        for i in range(len(l1)):
            result += np.abs(l1[i] - l2[i])
        result = result / len(l1)
        return result

    print("Over", len(_pred), "cars, the average gap between the predicted price and the real price is",
          round(average_gap(_pred, original), 0), "$")

    plt.figure(figsize=(15, 7))
    sns.displot(_pred, color="blue", label="Distrib Predictions")
    sns.displot(original, color="red", label="Distrib Original")
    plt.title("Distribution of pred and original Stage")
    plt.legend()
    return


def shap_visualization(_X_train, _model):
    shap.initjs()
    # Using a random sample of the dataframe for better time computation
    X_sampled = _X_train.sample(100, random_state=10)

    # explain the model's predictions using SHAP values
    # (same syntax works for LightGBM, CatBoost, and scikit-learn models)
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_sampled)

    # visualize the first prediction's explanation
    # shap.force_plot(explainer.expected_value, shap_values[0, :], X_sampled.iloc[0, :])

    # visualize the training set predictions
    # shap.force_plot(explainer.expected_value, shap_values, X_train)

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X_sampled, plot_type="bar")
    plt.savefig(os.path.join(ROOT_DIR, 'reports/figures/summary_shap.png'))
    return


if __name__ == '__main__':
    df = load_dataset()
    target = 'Stage'
    X_train, X_test, y_train, y_test = preprocessing_data(df, target)
    model = load_model('model.pkl')

    XGB_model_predict = model.predict(X_test)
    XGB_model_predict_proba = model.predict_proba(X_test)
    logger.info(classification_report(y_test, XGB_model_predict))

    shap_visualization(X_train, model)
    prediction_shap(model, df, XGB_model_predict, X_test)
