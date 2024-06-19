# Copyright (c) 2021-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tqdm.notebook import tqdm
import warnings
import json


class AuotencoderFeatureSelector:
    """
    A class to select features using an autoencoder, handling categorical and numerical data.
    Supports handling ambiguities in data types and ensures selected feature count constraints.

    Attributes:
        input_json (dict): List of dictionary objects to normalize into a dataframe
        id_column (str) : Column name that contains ID for morpheus AE pipeline. Default None.
        timestamp_column (str) : Column name that contains the
            timestamp for morpheus AE pipeline. Default None.
        encoding_dim (int): Dimension of the encoding layer, defaults to half of input
            dimensions if not set.
        batch_size (int): Batch size for training the autoencoder.
        variance_threshold (float) : Minimum variance a column must contain
            to remain in consideration. Default 0.
        null_threshold (float): Maximum proportion of null values a column can contain. Default 0.3.
        cardinality_threshold_high (float): Maximum proportion
            of cardinality to length of data allowable. Default 0.99.
        cardinality_threshold_low_n (int): Minimum cardinalty for a feature to be considered numerical
            during type infernce. Default 10.
        categorical_features (list[str]): List of features in the data to be considered categorical. Default [].
        numeric_features (list[str]): List of features in the data to be considered numeric. Default [].
        ablation_epochs (int): Number of epochs to train the autoencoder.
        device (str): Device to run the model on, defaults to 'cuda' if available.

    Methods:
        train_autoencoder(data_loader): Trains the autoencoder using the provided data loader.
        calculate_loss(model, data_loader): Calculates reconstruction loss using the trained model.
        preprocess_data(): Preprocesses the DataFrame to handle numerical and categorical data appropriately.
        remove_low_variance(processed_data): Removes features with low variance.
        remove_high_correlation(data): Removes highly correlated features based on a threshold.
        feature_importance_evaluation(processed_data): Evaluates feature importance using the autoencoder.
        select_features(k_min, k_max): Selects features based on importance, adhering to min/max constraints.
    """

    def __init__(
        self,
        input_json,
        id_column=None,
        timestamp_column=None,
        encoding_dim=None,
        batch_size=256,
        variance_threshold=0,
        null_threshold=0.3,
        cardinality_threshold_high=0.999,
        cardinality_threshold_low_n=10,
        categorical_features=[],
        numeric_features=[],
        ablation_epochs=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.df = pd.json_normalize(input_json)
        self.df_orig = pd.json_normalize(input_json)
        self.encoding_dim = encoding_dim or self.df.shape[1] // 2
        self.batch_size = batch_size
        self.device = device
        self.preprocessor = None  # To store the preprocessor for transforming data
        self.variance_threshold = variance_threshold
        self.null_threshold = null_threshold
        self.cardinality_threshold_high = cardinality_threshold_high
        self.cardinality_threshold_low = cardinality_threshold_low_n
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.ablation_epochs = ablation_epochs
        self.id_col = id_column
        self.ts_col = timestamp_column

        if self.id_col is not None:
            self.df.drop([self.id_col], axis=1, inplace=True)
            self.df_orig.drop([self.id_col], axis=1, inplace=True)

        if self.ts_col is not None:
            self.df.drop([self.ts_col], axis=1, inplace=True)
            self.df_orig.drop([self.ts_col], axis=1, inplace=True)

    class Autoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, encoding_dim), nn.ReLU(True)
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, input_dim), nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    def train_autoencoder(self, data_loader):
        """
        Trains the autoencoder model on the data provided by the DataLoader.

        Parameters
        __________
        data_loader: DataLoader)
            DataLoader containing the dataset for training.

        Returns
        ________
            The trained autoencoder model.
        """
        model = self.Autoencoder(
            data_loader.dataset.tensors[0].shape[1], self.encoding_dim
        ).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        model.train()
        for epoch in range(self.ablation_epochs):
            for inputs in data_loader:
                inputs = inputs[0].to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
        return model

    def calculate_loss(self, model, data_loader):
        """
        Calculates the reconstruction loss of the autoencoder model using the provided DataLoader.

        Parameters
        ___________
        model: Autoencoder
            The trained autoencoder model.

        data_loader: DataLoader)
            DataLoader for evaluating the model.

        Returns
        ________

        Mean reconstruction loss over the data in the DataLoader.
        """
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0].to(self.device)
                outputs = model(inputs)
                total_loss += nn.functional.mse_loss(outputs, inputs).item()
        return total_loss / len(data_loader)

    def preprocess_data(self):
        """
        Preprocesses the DataFrame by scaling numeric features and encoding categorical features.
        Handles ambiguities in data types by attempting to convert object types to numeric where feasible.
        Removes columns with high null values, high cardinality, or low cardinality.

        Returns
        _______

        The preprocessed data ready for feature selection.
        """
        # Parameters to define what constitutes high null, high cardinality, and low cardinality
        high_null_threshold = (
            self.null_threshold
        )  # Columns with more than 30% missing values
        high_cardinality_threshold = (
            self.cardinality_threshold_high
        )  # Columns with unique values > 50% of the total rows
        low_cardinality_threshold = (
            self.cardinality_threshold_low
        )  # Columns with fewer than 10 unique values

        print("\n##########\nPreprocessing Data")
        # Remove columns with high percentage of NULL values
        null_counts = self.df.isnull().mean()
        self.df = self.df.loc[:, null_counts <= high_null_threshold]
        col_uniqueness = {}
        # Remove columns with unhashable types
        for col in self.df.columns:

            try:
                unique_count = self.df[col].nunique()
                col_uniqueness[col] = unique_count
                total_count = self.df.shape[0]
            except TypeError:
                # Unhashable types
                self.df.drop(columns=[col], inplace=True)
                print(f"\t*Dropped unhashable column: {col}")
                continue

        dataframe_columns = list(self.df.columns)

        # Perform type inferencing if needed

        if self.categorical_features == [] or self.numeric_features == []:
            warnings.warn(
                "Categorical or numeric features not provided. Performing type inference which could be inaccurate."
            )

        if self.categorical_features == [] and self.numeric_features != []:
            self.categorical_features = [
                ft for ft in dataframe_columns if ft not in self.numeric_features
            ]

        elif self.categorical_features != [] and self.numeric_features == []:
            self.numeric_features = [
                ft for ft in dataframe_columns if ft not in self.categorical_features
            ]

        else:

            for col in self.df.columns:
                unique_count = col_uniqueness[col]
                total_count = self.df.shape[0]

                if unique_count < low_cardinality_threshold:
                    # Considered as categorical due to low cardinality
                    self.categorical_features.append(col)
                else:
                    # Try to convert 'object' columns to numeric if feasible
                    try:
                        self.df[col] = pd.to_numeric(self.df[col])
                        self.numeric_features.append(col)
                    except ValueError:
                        # Default to categorical if conversion fails
                        # Check cardinality
                        if unique_count / total_count > high_cardinality_threshold:
                            # Exclude from dataset due to high cardinality
                            print(f"\t*Dropped high cardinality column: {col}")
                            self.df.drop(columns=[col], inplace=True)
                        else:
                            self.categorical_features.append(col)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", OneHotEncoder(), self.categorical_features),
            ]
        )

        processed_data = self.preprocessor.fit_transform(self.df)

        # Convert to dense if the output is sparse
        if hasattr(processed_data, "toarray"):
            processed_data = processed_data.toarray()
            warnings.warn(
                "Found sparse arrays when one-hot encoding. Consider using fewer categorical variables."
            )

        self.preprocessed_df = pd.DataFrame(
            processed_data, columns=self.preprocessor.get_feature_names_out()
        )

        return processed_data

    def infer_column_types(self, df, sample_size=100, seed=None):
        """
        Infers the data types of all columns in a pandas DataFrame by sampling values from each column.

        Parameters
        __________
        df: pd.DataFrame
            The DataFrame whose column types are to be inferred.
        sample_size: int
            The number of samples to take from each column for type inference. Defaults to 100.
        seed: int
            An integer seed for the random number generator to ensure reproducibility of the sampling.

        Returns
        ________
        dict
            A dictionary mapping each column name to its inferred data type ('int', 'float', 'bool', or 'string').
        """

        type_dict = {}
        df = df.copy().infer_objects()

        for column in df.columns:
            col_type = str(df[column].dtype)
            if col_type.startswith("int"):
                type_dict[column] = "int"
            elif col_type.startswith("float"):
                type_dict[column] = "float"
            elif col_type.startswith("bool"):
                type_dict[column] = "bool"
            else:
                type_dict[column] = "string"

        return type_dict

    def prepare_schema(self, df, path=None):
        """
        Creates dictionary schema definition for use with Morpheus JSONSchemaBuilder.

        Dumps to json path if not None.

        Parameters
        __________
        df: pd.DataFrame
            Dataframe to generate schema for

        Returns
        _______
        scehma: dict
            Dictionary schema definition to dump to JSON
        """

        datatypes = self.infer_column_types(df)
        json_columns = list(
            set([col.split(".")[0] for col in df.columns if "." in col])
        )
        schema = {"JSON_COLUMNS": json_columns, "SCHEMA_COLUMNS": []}

        for column in datatypes.keys():

            if datatypes[column] != "bool":
                schema["SCHEMA_COLUMNS"].append(
                    {
                        "type": "ColumnInfo",
                        "dtype": datatypes[column],
                        "data_column": column,
                    }
                )
            else:
                schema["SCHEMA_COLUMNS"].append(
                    {
                        "type": "BoolColumn",
                        "dtype": "bool",
                        "data_column": column,
                        "name": column,
                    }
                )

        if self.id_col is not None:
            schema["SCHEMA_COLUMNS"].append(
                {"type": "ColumnInfo", "dtype": "string", "data_column": self.id_col}
            )

        if self.ts_col is not None:
            schema["SCHEMA_COLUMNS"].append(
                {
                    "type": "DateTimeColumn",
                    "dtype": "datetime",
                    "name": self.ts_col,
                    "data_column": self.ts_col,
                }
            )

        if path is not None:
            with open(path, "w") as f:
                json.dump(schema, f)

        return schema

    def remove_low_variance(self, processed_data):
        """
        Removes features with low variance from the dataset.

        Parameters
        ___________
        processed_data: np.array)
            Preprocessed data from which to remove low variance features.

        Returns
        ________
        tuple
            A tuple containing the reduced dataset and the mask of features retained (boolean array).
        """
        selector = VarianceThreshold(threshold=self.variance_threshold)
        reduced_data = selector.fit_transform(processed_data)
        print("\n##########\nDropped Features with Low Variance.")
        return reduced_data, selector.get_support()

    def remove_high_correlation(self, data, threshold=0.85):
        """
        Removes highly correlated features from the dataset based on the specified threshold.

        Parameters
        __________
        data: np.array
            The dataset from which to remove highly correlated features.

        threshold: float
            Correlation coefficient above which features are considered highly correlated and one is removed.

        Returns
        ________
        tuple
            A tuple containing the reduced dataset and a list of dropped columns.
        """
        corr_matrix = pd.DataFrame(data).corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_)
        )
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        print("\n##########\nDropped Features with High Correlation.")
        print(to_drop)
        return np.delete(data, to_drop, axis=1), to_drop

    def feature_importance_evaluation(self, processed_data):
        """
        Evaluates the importance of features by measuring the increase in the reconstruction loss of an
        autoencoder when each feature is individually omitted.

        Parameters
        ___________
        processed_data: np.array)
            The dataset to be used for assessing feature importance.

        Returns
        ________
        dict
            A dictionary where keys are feature indices and values are the calculated importance scores based on loss increase.
        """
        data_tensor = torch.FloatTensor(processed_data)
        data_loader = DataLoader(
            TensorDataset(data_tensor), batch_size=self.batch_size, shuffle=True
        )
        full_model = self.train_autoencoder(data_loader)
        base_loss = self.calculate_loss(full_model, data_loader)

        print("\n##########\nPerforming Autoencoder Ablation Study.")
        importance_scores = {}
        for i in tqdm(range(processed_data.shape[1]), desc="AE Ablation Study"):
            reduced_data = np.delete(processed_data, i, axis=1)
            reduced_tensor = torch.FloatTensor(reduced_data)
            reduced_loader = DataLoader(
                TensorDataset(reduced_tensor), batch_size=self.batch_size, shuffle=True
            )
            reduced_model = self.train_autoencoder(reduced_loader)
            loss = self.calculate_loss(reduced_model, reduced_loader)
            importance_scores[i] = base_loss - loss

        print("\nPerformed Autoencoder Ablation Study.")

        return importance_scores

    def print_report(self, feature_names):
        """
        Prints summary information on feature selection.

        Parameters
        __________
        feature_names: list[str]
            List of feature names that were downselected as important

        Returns
        _______
        list[str]
            List of feature names cleaned and processed for use.

        """

        cats = []
        nums = []

        for s in feature_names:
            if s.startswith("cat__"):
                cats.append(s[5:].split("_")[0])  # Strip 'cat__' and append
            elif s.startswith("num__"):
                nums.append(s[5:])  # Strip 'num__' and append

        cats = list(set(cats))
        print(
            f"\n##########\nThe following numeric features were found to be effective: "
        )
        print(nums)
        print(
            f"\n##########\nThe following categorical features were found to be effective: "
        )
        print(cats)

        return cats + nums

    def select_features(
        self, k_min=5, k_max=10, raw_schema_path=None, preprocess_schema_path=None
    ):
        """
        Selects features based on autoencoder performance, adhering to specified minimum and maximum feature count.

        Parameters
        ___________
        k_min: int
            Minimum number of features to retain.

        k_max: int
            Maximum number of features to retain.

        raw_schema_path: str
            Path to dump raw data schema file for Morpheus pipeline.

        preprocess_schema_path: str
            Path to dump preprocessed data schema file for Morpheus pipeline.

        Returns
        ________
        list
            selected features based on importance scores.
        """
        raw_schema = self.prepare_schema(self.df_orig, raw_schema_path)
        processed_data = self.preprocess_data()
        processed_data, support = self.remove_low_variance(processed_data)
        processed_data, dropped_indices = self.remove_high_correlation(processed_data)
        feature_scores = self.feature_importance_evaluation(processed_data)
        sorted_features = sorted(feature_scores, key=feature_scores.get, reverse=True)
        selected_features = sorted_features[: min(k_max, len(sorted_features))]
        final_features = selected_features[: max(k_min, len(selected_features))]
        final_feature_names = [
            f
            for i, f in enumerate(self.preprocessor.get_feature_names_out())
            if i in final_features
        ]
        final_feature_names = self.print_report(final_feature_names)
        preproc_schema = self.prepare_schema(
            self.df[final_feature_names], preprocess_schema_path
        )

        return raw_schema, preproc_schema
