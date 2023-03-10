import pandas as pd
import numpy as np
import os

from sklearn.impute import SimpleImputer, IterativeImputer, MissingIndicator, KNNImputer
from sklearn.neighbors import KNeighborsRegressor

class Cuttypanda:
    def __init__(self, filename):
        self.filename = filename
        self.extension = self.filename.split(".")[-1]
        self.data = None

    def __getattr__(self, attr):
        if attr == "shape":
            return self.get_shape()
        elif attr == "columns":
            return self.get_columns()
        elif attr == "dtypes":
            return self.get_dtypes()
        elif attr == "describe":
            return self.get_describe()
        elif attr == "missing_values":
            return self.get_missing_values()
        elif attr == "missing_rows":
            return self.get_missing_rows()
        elif attr == "unique_values":
            return self.get_unique_values()
        elif attr == "group_data":
            return self.group_data
        elif attr in [
            "mean_imputer",
            "median_imputer",
            "most_frequent_imputer",
            "constant_imputer",
        ]:
            return lambda: self.impute_missing_values(attr.split("_")[0])
        elif attr == "iterative_imputer":
            return self.iterative_imputer
        elif attr == "knn_imputer":
            return self.knn_imputer
        elif attr == "missing_indicator":
            return self.missing_indicator
        elif attr == "efficient_imputer":
            return self.efficient_imputer
        else:
            raise AttributeError(f"'Cuttypanda' object has no attribute '{attr}'")

    def load_data(self):
        if self.extension == "csv":
            self.data = pd.read_csv(self.filename)
        elif self.extension == "json":
            self.data = pd.read_json(self.filename)
        elif self.extension == "xlsx":
            self.data = pd.read_excel(self.filename)
        else:
            raise ValueError(
                "Invalid file extension. Only 'csv', 'json', and 'xlsx' are supported."
            )

    def get_shape(self):
        return self.data.shape

    def get_columns(self):
        return self.data.columns.tolist()

    def get_dtypes(self):
        return self.data.dtypes

    def get_describe(self):
        return self.data.describe()

    def get_missing_values(self):
        return self.data.isnull().sum().sum()

    def get_missing_rows(self):
        return self.data.isnull().any(axis=1).sum()

    def get_unique_values(self):
        unique_vals = {}
        for col in self.data.columns:
            unique_vals[col] = self.data[col].nunique()
        return unique_vals

    def group_data(self, group_cols, agg_func):
        grouped_data = self.data.groupby(group_cols).agg(agg_func)
        return grouped_data

    def impute_missing_values(self, strategy):
        imputer = SimpleImputer(strategy=strategy)
        imputed_data = imputer.fit_transform(self.data)
        return imputed_data

    def iterative_imputer(self, **kwargs):
        imputer = IterativeImputer(**kwargs)
        imputed_data = imputer.fit_transform(self.data)
        return imputed_data

    def knn_imputer(self, **kwargs):
        imputer = KNNImputer(**kwargs)
        imputed_data = imputer.fit_transform(self.data)
        return imputed_data

    def missing_indicator(self, **kwargs):
        indicator = MissingIndicator(**kwargs)
        missing_data = indicator.fit_transform(self.data)
        return missing_data

    def clean_data(self) -> None:
        # Drop any rows with missing values
        self.data.dropna(inplace=True)

        # Remove any whitespace from column names
        self.data.rename(columns=lambda x: x.strip(), inplace=True)

        # Convert any string columns to lowercase
        string_cols = self.data.select_dtypes(include=["object"]).columns
        self.data[string_cols] = self.data[string_cols].apply(lambda x: x.str.lower())

        # Remove any leading/trailing whitespace from string columns
        self.data[string_cols] = self.data[string_cols].apply(lambda x: x.str.strip())

        # Convert any date/time columns to datetime format
        date_cols = self.data.select_dtypes(include=["datetime64"]).columns
        self.data[date_cols] = self.data[date_cols].apply(
            lambda x: pd.to_datetime(x, errors="coerce")
        )

        self.data.drop_duplicates(inplace=True)

        self.data.reset_index(drop=True, inplace=True)

    def get_head(self, n_row) -> pd.DataFrame:
        if n_row > 0:
            return self.data.head(n_row)
        raise ValueError("Unsupported value for row")

    def explore_data(self):
        print(self.data.describe())

        print(self.data.head())

        print(self.data.info())

        corr_matrix = self.data.corr()
        print(corr_matrix)

        pd.plotting.scatter_matrix(self.data, figsize=(10, 10))

    def plot_data(self, x_col, y_col):
        self.data.plot(kind="scatter", x=x_col, y=y_col, figsize=(8, 8))

    def group_data(self, group_cols, agg_func):
        grouped_data = self.data.groupby(group_cols).agg(agg_func)

        sort_col = list(grouped_data.columns)[0]
        grouped_data = grouped_data.sort_values(sort_col, ascending=False)

        print(grouped_data)

    def efficient_imputer(self, k_neighbors=5, max_iter=100, tol=1e-3):
        # Create a copy of the data
        imputed_data = self.data.copy()

        # Identify missing values
        missing_mask = np.isnan(imputed_data)

        # Create a KNN regressor to impute missing values
        knn = KNeighborsRegressor(n_neighbors=k_neighbors)

        # Loop through each column with missing values
        for col in range(imputed_data.shape[1]):
            # Identify the missing values in this column
            col_missing_mask = missing_mask[:, col]

            # If there are missing values, impute them using KNN
            if col_missing_mask.sum() > 0:
                # Create an array of indices for the rows with missing values
                missing_indices = np.where(col_missing_mask)[0]

                # Create an array of indices for the rows without missing values
                not_missing_indices = np.where(~col_missing_mask)[0]

                # Get the values of the column for the rows without missing values
                not_missing_values = imputed_data[not_missing_indices, col]

                # If all the rows in the column are missing, skip it
                if not_missing_indices.shape[0] == 0:
                    continue

                # Fit the KNN regressor to the non-missing values
                knn.fit(not_missing_indices.reshape(-1, 1), not_missing_values)

                # Use the KNN regressor to predict the missing values
                predicted_values = knn.predict(missing_indices.reshape(-1, 1))

                # Replace the missing values with the predicted values
                imputed_data[missing_indices, col] = predicted_values

        # Return the imputed data
        return imputed_data

    def mean_imputer(self):
        imputed_data = self.data.copy()
        imputed_data.fillna(imputed_data.mean(), inplace=True)
        return imputed_data

    def median_imputer(self):
        imputed_data = self.data.copy()
        imputed_data.fillna(imputed_data.median(), inplace=True)
        return imputed_data

    def most_frequent_imputer(self):
        imputed_data = self.data.copy()
        imputed_data.fillna(imputed_data.mode().iloc[0], inplace=True)
        return imputed_data
