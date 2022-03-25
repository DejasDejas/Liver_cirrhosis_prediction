# -*- coding: utf-8 -*-
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
import logging
from config.config import processed_path, raw_path
from src.data.treatments import Treatment

logger: logging.Logger = logging.getLogger(__name__)


def yes_or_no(_question):
    while "the answer is invalid":
        reply = str(input(_question+' (Y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False
        print('please enter a valid response (Y/n)')


def get_raw_df_name():
    """
    Return the raw dataset name if exist.
    """
    print(raw_path, processed_path)
    _file_name = next(os.walk(raw_path), (None, None, []))[2]
    assert _file_name, "They're not dataset in raw folder."
    return _file_name[0]


def save_dataframe(_df, file_name=None):
    if file_name is None:
        file_name = get_raw_df_name()  # by default, get the raw file name
    _df.to_csv(os.path.join(processed_path, file_name), index=False)
    logger.info('Dataframe {} saved in processed folder !'.format(file_name))
    return


def load_dataset(df_name=None, raw=False):
    """
    Load dataframe from processed folder.
    If this folder empty, we ask to copy it and load it.
    If file doesn't copy or exist, return None.
    """
    if df_name is None:
        df_name: str = get_raw_df_name()
    df_name_path = os.path.join(processed_path, df_name)

    if raw:
        df_name = get_raw_df_name()
        df_name_path = os.path.join(raw_path, df_name)
    elif not os.path.exists(df_name_path):
        logger.warning("There are not the {} df to load in the 'processed' folder !".format(df_name))
        logger.error("You must run 'make_dataset' script before load df!")
        return None

    data_loaded = pd.read_csv(df_name_path)
    logger.info("Data '{}' loaded with shape: {}".format(df_name, data_loaded.shape))
    return data_loaded


def df_treatment(df, median=True, mean=False, outliers=False):
    treat = Treatment(df, 'Stage')
    treat.age_days_to_years()  # replace days unity for Age to year
    treat.missing_values(median=median, mean=mean)
    treat.delete_column(['ID'])  # remove 'id' feature
    treat.delete_column(['Status'])
    treat.delete_column(['N_Days'])
    treat.to_categorical()
    if outliers:
        outliers_columns_list = df.select_dtypes(
            include=['int64', 'float64']).columns.drop(['Stage', 'Age'])
        for col in outliers_columns_list:
            treat.remove_outliers(col, replace_by_min_max=True)
    return df


def make_nan_mean_df():
    df_name = 'Nan_mean_cirrhosis.csv'
    dataframe = load_dataset(raw=True)
    df_nan_mean = dataframe.copy()
    df_nan_mean = df_treatment(df_nan_mean, median=False, mean=True)
    save_dataframe(df_nan_mean, df_name)
    logger.info("Nan mean df was create and saved !")
    return


def make_outliers_df():
    df_name = 'outliers_cirrhosis.csv'
    dataframe = load_dataset(raw=True)
    df_outliers = dataframe.copy()
    df_outliers = df_treatment(df_outliers, outliers=True)
    save_dataframe(df_outliers, df_name)
    logger.info("outliers df was create and saved !")
    return


def make_default_df():
    dataframe = load_dataset(raw=True)
    dataframe = df_treatment(dataframe)
    save_dataframe(dataframe)
    logger.info("default df was create and saved !")
    return


def existing_make_df(_df_name):
    df_name_path = os.path.join(processed_path, _df_name)
    if os.path.exists(df_name_path):
        logger.warning("There is already a dataframe {}".format(_df_name))
        question = "Do you want to replace the existing dataframe?"
        answer = yes_or_no(question)
        if answer:
            os.remove(df_name_path)
            logger.info("The existing {} dataset was deleted.".format(_df_name))
            return False
        else:
            logger.error("No action. \n To load the existing dataset {} run 'load_dataset'.".format(_df_name))
            return True
    return False


def main(default=True, nan_mean=False, outliers=False):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    Preprocessing data raw to make cleaned dataset.
    """
    if default:
        df_name = 'cirrhosis.csv'
        if not existing_make_df(df_name):
            make_default_df()
    if nan_mean:
        nan_mean_df_name = 'Nan_mean_cirrhosis.csv'
        if not existing_make_df(nan_mean_df_name):
            make_nan_mean_df()
    if outliers:
        outliers_df_name = 'outliers_cirrhosis.csv'
        if not existing_make_df(outliers_df_name):
            make_outliers_df()
    return


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(nan_mean=True, outliers=True)
