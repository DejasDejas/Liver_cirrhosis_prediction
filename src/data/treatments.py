from sklearn.model_selection import train_test_split
import logging

logger: logging.Logger = logging.getLogger(__name__)


class Treatment:
    """
    This class to effectuate data treatments.
    Params:
    - dataframe: pandas.DataFrame, raw data
    - target_feature: string, feature to predict name
    """

    def __init__(self, dataframe, target):
        """
        Attributes initialization
        """
        if target in dataframe.columns:
            self.dataframe = dataframe  # dataframe.copy()
            self.target = target
        else:
            raise ValueError("Target feature doesn't exist in dataframe.")

    def column_exist(self, column):
        """
        Function to check if a column exists.
        """
        return column in self.dataframe.columns

    def delete_column(self, column):
        """
        Function delete a column.
        params:
        - column: list, list of column to remove.
        """
        assert isinstance(column, list), "column variable must be a list !"
        for col in column:
            assert self.column_exist(col), "{} column doesn't exist.".format(column)
            self.dataframe.drop(col, axis=1, inplace=True)
        return

    def age_days_to_years(self):
        """
        Transform Age value from days to years
        """
        self.dataframe['Age'] = (self.dataframe['Age']/365).astype('int64')
        return

    def to_categorical(self):
        """
        This function allows a category to be modified by another category within a categorical variable.
        """
        def label_encoding(label):
            from sklearn import preprocessing
            label_encoder = preprocessing.LabelEncoder()
            self.dataframe[label] = label_encoder.fit_transform(self.dataframe[label])
            self.dataframe[label].unique()
        categorical_columns = self.dataframe.select_dtypes('object').columns
        for column in categorical_columns:
            label_encoding(column)
        return

    def remove_outliers(self, column, replace_by_min_max=False, replace_by_median=False):
        """
        This function remove outliers.
        """
        if self.column_exist(column):
            quantile_1 = self.dataframe[column].quantile(0.25)
            quantile_3 = self.dataframe[column].quantile(0.75)
            median = self.dataframe[column].median()
            inter_quantile = quantile_3 - quantile_1
            down_limit = quantile_1 - 1.5 * inter_quantile
            up_limit = quantile_3 + 1.5 * inter_quantile

            if replace_by_median:
                def replace_outliers(x): return median if (x < down_limit or x > up_limit) else x
                self.dataframe[column] = self.dataframe[column].apply(replace_outliers)
            elif replace_by_min_max:
                self.dataframe.loc[self.dataframe[column] > up_limit, column] = up_limit
                self.dataframe.loc[self.dataframe[column] < down_limit, column] = down_limit
            else:
                self.dataframe = self.dataframe[(self.dataframe[column] > down_limit) &
                                                (self.dataframe[column] < up_limit)]
        else:
            raise ValueError(f"{column} column doesn't exist.")
        return

    def split_dataset(self, dataframe, test_size=0.2):
        """
        This function split dataset into training data and test data
        """
        # y = dataframe[self.target].values
        y = dataframe[self.target]
        # X = dataframe.drop(self.target, axis=1).values
        X = dataframe.drop(self.target, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=4, shuffle=True)

        return X_train, X_test, y_train, y_test

    def encoding(self):
        """
        This function allows the encoding of qualitative values
        """
        dataframe = self.dataframe.copy()
        for col in dataframe.select_dtypes("category").columns:
            dataframe[col] = self.dataframe[col].cat.codes
        return dataframe

    def missing_values(self, median=True, mean=False):
        """
        replace missing value by feature mean for numerical features and by
        most occurrence value fo categorical values.
        """
        assert median != mean, "You must choose median or mean to fill in the nan values, not both or neither."

        df_num_col = self.dataframe.select_dtypes(include=['int64', 'float64']).columns
        df_cat_col = self.dataframe.select_dtypes(include='object').columns

        for col in df_num_col:
            if median:
                self.dataframe[col].fillna(self.dataframe[col].median(), inplace=True)
            elif mean:
                self.dataframe[col].fillna(self.dataframe[col].mean(), inplace=True)
            else:
                raise TypeError('At least one of the two values \'mean\' or \'median\' must be chosen !')

        for col in df_cat_col:
            self.dataframe[col].fillna(self.dataframe[col].mode().values[0], inplace=True)
        return
