from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import MAPPING_DATA_FILENAME


class VolumeNormalizer(BaseEstimator, TransformerMixin):
    """Volume Normalizer"""

    def __init__(self, column: str = None, mapping: Dict[str, float] = None, save_mapping: bool = True):
        super().__init__()
        self.column = column
        self.mapping = mapping
        self.save = save_mapping

    def fit(self, X: pd.DataFrame, y=None):  # pylint: disable=unused-argument
        self.mapping = X.query("month_num == -1").set_index(["country", "brand"]).loc[:, self.column].to_dict()
        if self.save:
            self._save_mapping()
        return self

    def transform(self, X: pd.DataFrame, y=None):  # pylint: disable=unused-argument
        Xt = X.set_index(["country", "brand"]).loc[:, "volume"]
        Xt = Xt.divide(Xt.index.map(self.mapping))
        return Xt.values

    def inverse_transform(self, X_inv: pd.DataFrame, col=None):
        X = X_inv.set_index(["country", "brand"]).loc[:, col]
        X = X.multiply(X.index.map(self.mapping))
        return X.values

    def _save_mapping(self):
        print(f"Saving mapping into: {MAPPING_DATA_FILENAME}")
        dump(self.mapping, MAPPING_DATA_FILENAME)


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """Cyclical Encoder
    This class encodes month labels into sinus and cosinus.
    """

    def __init__(self, freq: float = 12, drop_cols: bool = False):
        super().__init__()
        self.month_list: List[str] = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        self.mapping: Dict[str, int] = {m: i + 1 for i, m in enumerate(self.month_list)}
        self.columns: List[str] = list()
        self.freq = freq
        self.drop_cols = drop_cols

    def fit(self, X: pd.DataFrame, y=None):  # pylint: disable=unused-argument
        self.columns = X.columns.to_list()
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:  # pylint: disable=unused-argument
        for c in self.columns:
            values = X[c].map(self.mapping)
            X[c + "_sin"] = np.sin(2 * np.pi * values / self.freq)
            X[c + "_cos"] = np.cos(2 * np.pi * values / self.freq)
            if self.drop_cols:
                X = X.drop(c, axis=1)
        return X

    def get_feature_names(self, input_features: List[str] = None):  # pylint: disable=unused-argument
        """
        Return feature names for output features.
        Parameters
        ----------
        input_features : list of str of shape (n_features,)
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.
        Returns
        -------
        output_feature_names : ndarray of shape (n_output_features,)
            Array of feature names.
        """
        feature_names = []
        for f in self.columns:
            if not self.drop_cols:
                feature_names.extend([f])
            feature_names.extend([f + "_sin", f + "_cos"])

        return np.array(feature_names, dtype=object)


def get_feature_names_(column_transformer, all_columns) -> List[str]:
    #   all_columns = X.columns
    columns_processed = []
    for step in column_transformer.transformers_:
        step_name, step_obj, step_vars = step
        print(step_name)
        # print(step_obj)
        if hasattr(step_obj, "get_feature_names"):
            vars_ = column_transformer.named_transformers_[step_name].get_feature_names(input_features=step_vars)
            columns_processed.extend(vars_)
            print(vars_)
        elif step_name == "remainder" and step_obj == "passthrough":
            print("remainder")
            vars_ = [all_columns[i] for i in step_vars]
            columns_processed.extend(vars_)
            print(vars_)
        else:
            columns_processed.extend(step_vars)
            print(step_vars)
    return columns_processed
