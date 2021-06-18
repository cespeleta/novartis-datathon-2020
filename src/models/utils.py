import re
from typing import Any, Dict, Optional

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer  # SimpleImputer,

# from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.fixes import loguniform

from src.features.encoders import CyclicalEncoder

from sklearn.experimental import enable_hist_gradient_boosting  # isort:skip
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # isort:skip


class loguniform_int:
    """Integer valued version of the log-uniform distribution"""

    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)


def get_model_name(pipe) -> str:
    """Return the model name"""
    # https://stackoverflow.com/questions/14596884/remove-text-between-and
    model_name = pipe.named_steps.get("regressor").__class__.__name__
    model_name = re.sub(r"[\(\[].*?[\)\]]", "", model_name)
    return model_name


def get_preprocessors() -> ColumnTransformer:
    # https://iaml.it/blog/optimizing-sklearn-pipelines

    # Declare encoders
    ordinal_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=int)
    cyclical_encoder = CyclicalEncoder(drop_cols=True)
    # simple_imputer = SimpleImputer(missing_values=np.nan, add_indicator=False, strategy="median")
    knn_imputer = KNNImputer(n_neighbors=3, weights="uniform")

    # tree_preproc_1 = ColumnTransformer(
    #     [
    #         (
    #             "ordinal-encoder",
    #             ordinal_preprocessor,
    #             ["country", "brand", "therapeutic_area", "presentation", "month_name"],
    #         )
    #     ],
    #     remainder="passthrough",
    # )

    # tree_preproc_2 = ColumnTransformer(
    #     [
    #         ("ordinal-encoder", ordinal_preprocessor, ["country", "brand", "therapeutic_area", "presentation"]),
    #         ("cyclical-encoder", cyclical_encoder, ["month_name"]),
    #         ("simple-imputer", simple_imputer, make_column_selector(pattern="lag_")),
    #     ],
    #     remainder="passthrough",
    # )

    tree_preproc_3 = ColumnTransformer(
        [
            ("ordinal-encoder", ordinal_preprocessor, ["country", "brand", "therapeutic_area", "presentation"]),
            ("cyclical-encoder", cyclical_encoder, ["month_name"]),
            ("knn-imputer", knn_imputer, make_column_selector(pattern="lag_")),
        ],
        remainder="passthrough",
    )

    return tree_preproc_3


def get_models(name: str) -> Optional[Dict[str, Any]]:

    __all__ = ["HistGradientBoostingRegressor", "RandomForestRegressor"]
    if name not in __all__:
        ValueError(f"{name} is not a correct value")

    models = dict()
    models["HistGradientBoostingRegressor"] = {
        "estimator": HistGradientBoostingRegressor(max_iter=10, max_depth=3, random_state=0),
        "params": {
            "regressor__l2_regularization": loguniform(1e-6, 1e3),
            "regressor__learning_rate": loguniform(0.001, 10),
            "regressor__max_leaf_nodes": loguniform_int(2, 256),
            "regressor__min_samples_leaf": loguniform_int(1, 100),
            "regressor__max_depth": loguniform_int(3, 17),
            "regressor__max_iter": loguniform_int(20, 200),
            "regressor__max_bins": loguniform_int(2, 255),
        },
    }
    models["RandomForestRegressor"] = {
        "estimator": RandomForestRegressor(n_estimators=10, max_depth=3, random_state=0),
        "params": {
            "regressor__n_estimators": loguniform_int(20, 200),
            "regressor__max_depth": loguniform_int(3, 17),
            "regressor__min_samples_split": loguniform_int(5, 100),
            "regressor__min_samples_leaf": loguniform_int(5, 100),
            "regressor__max_features": ["auto", "sqrt", "log2", None],
            "regressor__min_impurity_decrease": [0.0, 0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 1.0],
        },
    }

    if name is None:
        return models
    return models.get(name)
