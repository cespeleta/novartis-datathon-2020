import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from src.config import PATH_DATA_PROCESSED, PATH_MODELS
from src.features.build_features import get_products_for_train
from src.metrics.metrics import coverage_fraction_score, show_metrics
from src.models.utils import get_model_name, get_models, get_preprocessors
from src.utils import get_logger, logger_args

logger = get_logger("train_model")

VERBOSE = 1  # level of verbose: 0, 1, 2
SCORING = {
    "MAE": "neg_mean_absolute_error",
    "MAPE": "neg_mean_absolute_percentage_error",
    "COV": make_scorer(coverage_fraction_score),
}

# Define fold strategy
outer_cv = GroupKFold(n_splits=5)   # for generalization
inner_cv = GroupKFold(n_splits=20)  # for model hyper-param selection


def evaluate_model(estimator, X: pd.DataFrame, y: pd.Series, groups=None, cv=None):
    cv_results = cross_validate(
        estimator,
        X,
        y,
        groups=groups,
        scoring=SCORING,
        cv=cv,
        verbose=VERBOSE,
        fit_params={"groups": groups},
        return_train_score=True,
        return_estimator=True,
        n_jobs=-1,
    )
    logger.info("Training process took: {:.2f} mins".format(cv_results.get("fit_time").sum() / 60))
    show_metrics(cv_results, SCORING)
    return cv_results


def get_predictions(estimators, X: pd.DataFrame, y: pd.Series, groups=None, cv=None) -> np.ndarray:
    """Out of fold predictions"""
    # get estimators from the `cross_validate` function
    predictions = np.zeros(X.shape[0], dtype=float)
    for estimator, (_, test_idx) in zip(estimators, cv.split(X, y, groups=groups)):
        predictions[test_idx] = estimator.predict(X.iloc[test_idx])
    return predictions


def get_intervals(y_true: np.ndarray, y_pred: np.ndarray, mult: float = 1.96):
    """Confidence Intervals"""
    std = np.subtract(y_true, y_pred).std() * mult
    return y_pred - std, y_pred + std, std


def make_predictions_df(
    y_true: np.ndarray, y_low: np.ndarray, y_pred: np.ndarray, y_high: np.ndarray, index: List[str] = None
) -> pd.DataFrame:
    columns = ["y_true", "y_low", "y_pred", "y_high"]
    preds_df = pd.DataFrame(np.column_stack([y_true, y_low, y_pred, y_high]), columns=columns)
    if index is not None:
        preds_df.set_index(index, inplace=True)
    return preds_df


def train_model(estimator, X: pd.DataFrame, y: pd.Series, cv) -> Dict[str, Any]:  # target_col cv=outer_cv
    """Cross validate an estimator"""
    # Tune hyper-parameters
    outer_groups = list(X.country + ", " + X.brand)
    cv_results = evaluate_model(estimator, X, y, outer_groups, cv)

    # Get out-of-fold predictions
    estimators = cv_results.get("estimator")
    y_pred = get_predictions(estimators, X, y, groups=outer_groups, cv=cv)

    # Calculate confidence intervals based on the standard
    # deviation of the residuals
    y_low, y_high, std = get_intervals(y, y_pred)

    # Prepare output with predicted values and confidence intervals
    preds_df = make_predictions_df(y, y_low, y_pred, y_high, index=[X.country, X.brand])

    # Prepare output for cross validation scores
    scores_df = pd.DataFrame(cv_results).drop("estimator", axis=1)

    # log best params
    for e in estimators:
        logger.info(e.best_params_)

    # Save model information
    to_save = dict()
    to_save["models"] = estimators
    to_save["features"] = X.columns.to_list()
    to_save["predictions"] = preds_df
    to_save["cv_results"] = scores_df
    to_save["conf_interval_std"] = std
    return to_save


def train_models(estimator, target_cols: List[str], dataset: Path, cv) -> Path:
    """Train all the models"""
    output_path = PATH_MODELS / dataset / get_model_name(estimator.estimator)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model information in: {output_path}")

    all_preds = dict()
    all_scores = dict()
    for target_col in target_cols:
        # Load train data
        data = load(PATH_DATA_PROCESSED / dataset / f"features_for_{target_col}.joblib")
        X, y = data.get("X"), data.get("y")

        # Only train with products that have values in the target
        mask = get_products_for_train(y)
        X, y = X.loc[mask, :], y.loc[mask, :]

        logger.info(f"- Training model {target_col} with {X.shape[0]} products")
        model_dict = train_model(estimator, X, y, cv=cv)
        dump(model_dict, output_path / f"model_{target_col}.joblib")

        all_preds[target_col] = model_dict.get("predictions")
        all_scores[target_col] = model_dict.get("cv_results")

    # Overall performance and store information
    print("Overall performance")
    show_metrics(pd.concat(all_scores), SCORING)

    # Competition metric
    if len(target_cols) == 24:
        # store all out-of-fold-predictions and scores
        dump(all_preds, output_path / "all_preds.joblib")
        dump(all_scores, output_path / "all_scores.joblib")

    return output_path


def main():

    # Evaluate multiple regressors
    # https://iaml.it/blog/optimizing-sklearn-pipelines
    # https://stackoverflow.com/questions/38555650/try-multiple-estimator-in-one-grid-search/53292354

    # read parameters from command line
    args = _parse_args()

    # get pre-processor
    preprocessor = get_preprocessors()
    # get estimator and hyper-params
    regressor_dict = get_models(args.model_class)
    regressor = regressor_dict.get("estimator")
    param_distributions = regressor_dict.get("params")
    # define the pipeline: pre-processor + estimator
    model = Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])
    # run randomized search over the pipeline
    model_random_search = RandomizedSearchCV(
        model,
        refit="MAE",
        n_jobs=1,
        verbose=VERBOSE,
        param_distributions=param_distributions,
        n_iter=args.n_iter_search,
        cv=inner_cv,
        scoring=SCORING,
    )
    # execute the cross-validation for all the 24 models (0 - 23)
    _ = train_models(model_random_search, args.target_cols, args.dataset, cv=outer_cv)


def _parse_args():
    parser = argparse.ArgumentParser()

    all_targets = [f"y_{i}" for i in range(24)]
    parser.add_argument(
        "--target_cols",
        type=str,
        nargs="+",
        default=all_targets,
        help="Targets that we want to train the model to [y_0, ..., y_23].",
    )
    parser.add_argument(
        "--dataset", type=str, default="win_len_24", help="Dataset that we want to use to train the model"
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="HistGradientBoostingRegressor",
        help="Choose between `HistGradientBoostingRegressor` or `RandomForestRegressor`",
    )
    parser.add_argument(
        "--n_iter_search",
        type=int,
        default=3,
        help=(
            "Number of parameter settings that are sampled. `n_iter_search` trades off "
            "runtime vs quality of the solution"
        ),
    )
    # Log parameters
    logger.info("Settings used to train the models:")
    logger_args(logger, parser)
    return parser.parse_args()


if __name__ == "__main__":

    main()

    # _parse_args()
