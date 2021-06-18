import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.pipeline import Pipeline

from src.config import (
    INDEX_COLS,
    MAPPING_DATA_FILENAME,
    PATH_DATA_PROCESSED,
    PATH_MODELS,
)
from src.datasets.generics_dataset import GenericsDataset
from src.features.build_features import get_products_for_train, series_to_supervised
from src.features.encoders import VolumeNormalizer
from src.utils import get_logger, logger_args

logger = get_logger("predict_model")


class ForecastData:
    def __init__(self, models_folder, features_folder, volume_df, metadata_df, submission_template_df):
        self.models_folder = models_folder
        self.features_folder = features_folder
        self.volume_df = volume_df
        self.metadata_df = metadata_df
        self.submission_template_df = submission_template_df

        self.predictions = None
        self.pred_95_low = None
        self.pred_95_high = None

        # Load mapping and instantiate VolumeNormalizer
        self.mapping = load(MAPPING_DATA_FILENAME)
        self.scaler = VolumeNormalizer(column="volume", mapping=self.mapping)

    def _prepare_submission_datasets(self) -> None:
        # Convert volume to wide format and keep targets y_0, y_1, ..., y_23
        self.volume_df["volume_normalized"] = self.scaler.transform(self.volume_df)
        gx_volume_w = series_to_supervised(self.volume_df, col="volume_normalized")
        gx_volume_w = gx_volume_w.loc[:, gx_volume_w.columns.str.startswith("y_")]

        # Prepare datasets to predict future data. In some products, we have data in
        # in y_1, y_2, etc. we should use it in order to get better performance
        self.predictions = (
            self.submission_template_df.loc[:, INDEX_COLS]
            .drop_duplicates()
            .merge(gx_volume_w, on=INDEX_COLS, how="left")
            .set_index(INDEX_COLS)
        )

        self.pred_95_low = (
            self.submission_template_df.loc[:, INDEX_COLS + ["month_num", "pred_95_low"]]
            .pivot(index=INDEX_COLS, columns="month_num", values="pred_95_low")
            .add_prefix("y_")
        )

        self.pred_95_high = (
            self.submission_template_df.loc[:, INDEX_COLS + ["month_num", "pred_95_high"]]
            .pivot(index=INDEX_COLS, columns="month_num", values="pred_95_high")
            .add_prefix("y_")
        )
        return None

    def _merge_dataframes(self):
        predictions_ = (
            self.predictions.reset_index()
            .melt(id_vars=INDEX_COLS, var_name="month_num", value_name="prediction")
            .assign(month_num=lambda x: x.month_num.str.replace("y_", "").astype(int))
            .sort_values(by=INDEX_COLS + ["month_num"])
        )

        pred_95_low_ = (
            self.pred_95_low.reset_index()
            .melt(id_vars=INDEX_COLS, var_name="month_num", value_name="pred_95_low")
            .assign(month_num=lambda x: x.month_num.str.replace("y_", "").astype(int))
            .sort_values(by=INDEX_COLS + ["month_num"])
        )

        pred_95_high_ = (
            self.pred_95_high.reset_index()
            .melt(id_vars=INDEX_COLS, var_name="month_num", value_name="pred_95_high")
            .assign(month_num=lambda x: x.month_num.str.replace("y_", "").astype(int))
            .sort_values(by=INDEX_COLS + ["month_num"])
        )

        df_preds = pred_95_low_.merge(predictions_, on=["country", "brand", "month_num"]).merge(
            pred_95_high_, on=["country", "brand", "month_num"]
        )
        return df_preds

    def _forecast_to_original_scale(self, submission: pd.DataFrame) -> pd.DataFrame:
        submission["pred_95_low"] = self.scaler.inverse_transform(submission, col="pred_95_low")
        submission["forecast"] = self.scaler.inverse_transform(submission, col="prediction")
        submission["pred_95_high"] = self.scaler.inverse_transform(submission, col="pred_95_high")
        return submission

    def make_predictions(self) -> pd.DataFrame:

        # Create dataframes for submission
        self._prepare_submission_datasets()

        # Load features for all models and keep only those that we are
        # interested in predict the future (191 products from the submission file)
        data = load(self.features_folder / "features_for_y_0.joblib")
        X = data.get("X").merge(self.predictions.reset_index(), on=INDEX_COLS, how="right")
        logger.info(f"Number of products to forecast: {X.shape[0]}")

        for i in range(24):
            target_col = f"y_{i}"
            logger.info(f"Predicting values for: {target_col}")

            # Load model
            mdl_name = f"model_{target_col}.joblib"
            mdl_obj = load(self.models_folder / mdl_name)

            # Get objects from the dict
            mdls = mdl_obj.get("models")
            features = mdl_obj.get("features")
            conf_int_std = mdl_obj.get("conf_interval_std")

            # Predict points values
            y_pred = np.mean([mdl.predict(X.loc[:, features]) for mdl in mdls], axis=0)
            y_true = X.loc[:, target_col]

            mask = y_true.notnull()
            X.loc[:, target_col] = y_true.where(mask, y_pred).values
            self.predictions.loc[:, target_col] = y_true.where(mask, y_pred).values
            self.pred_95_low.loc[:, target_col] = self.predictions.loc[:, target_col].values - conf_int_std
            self.pred_95_high.loc[:, target_col] = self.predictions.loc[:, target_col].values + conf_int_std

        # Combine all 3 DataFrames for submission
        submission_df = self._merge_dataframes()

        # Inverse transform predictions to original scale
        submission_df = self._forecast_to_original_scale(submission_df)

        # Save predictions into a file
        output_file = self.models_folder / "forecast_predictions.joblib"
        logger.info(f"Saving forecast predictions in: {output_file}")
        dump(submission_df, output_file)
        return submission_df


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", type=str, default="win_len_24", help="Dataset that we want to use to train the model"
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="HistGradientBoostingRegressor",
        help="Choose between `HistGradientBoostingRegressor` or `RandomForestRegressor`",
    )

    # Log parameters
    logger.info("Settings used to predict:")
    logger_args(logger, parser)
    return parser.parse_args()


def main():

    # read parameters from command line
    args = _parse_args()

    # Load dataset
    generics = GenericsDataset()
    volume_df = generics["gx_volume"]
    metadata_df = generics.get_metadata()
    submission_template_df = generics["submission_template"]

    # Model list file path
    models_folder = PATH_MODELS / args.dataset / args.model_class
    features_folder = PATH_DATA_PROCESSED / args.dataset

    f_class = ForecastData(models_folder, features_folder, volume_df, metadata_df, submission_template_df)
    submission_df = f_class.make_predictions()
    # logger.info(submission_df)


if __name__ == "__main__":
    main()
