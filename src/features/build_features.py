import argparse
from typing import List

import numpy as np
import pandas as pd
from joblib import dump

from src.config import PATH_DATA_PROCESSED
from src.datasets.generics_dataset import GenericsDataset
from src.features.encoders import VolumeNormalizer
from src.utils import get_logger, logger_args

logger = get_logger("build_features")

# def get_products_for_train(data: pd.DataFrame, target_col: str = "y_0") -> np.ndarray:
#     """Pass volume data and detect products used to train"""
#     return data.loc[:, target_col].notnull().values


def get_products_for_train(s: pd.Series) -> np.ndarray:
    """Pass volume data and detect products used to train"""
    return s.notnull().values


def series_to_supervised(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Pivot dataframe

    Args:
        df (pd.DataFrame): Data in long format
        col (str): Column used in values

    Returns:
        pd.DataFrame: [description]

    Examples:
        # Create mock data frame
        df = pd.DataFrame({
            "country": ["country_1"] * 12,
            "brand": ["brand_1"] * 12,
            "month_num": list(range(12)),
            "volume": list(range(12))
        })
        print(series_to_supervised(df, col="volume"))
        #                     y_0  y_1  y_2  y_3  y_4  y_5  y_6  y_7  y_8  y_9  y_10  y_11
        # country   brand
        # country_1 brand_1    0    1    2    3    4    5    6    7    8    9    10    11
    """

    df_wide = (
        df.pivot(values=col, index=["country", "brand"], columns="month_num")
        # .fillna(method="backfill", axis=1)
    )
    df_wide.columns = [f"lag_{abs(c)}" if c < 0 else f"y_{c}" for c in df_wide.columns]
    logger.info(df_wide.shape)
    return df_wide


def get_columns_range(data_wide: pd.DataFrame, target_col: str, win_len: int) -> List[str]:
    """Select range of columns based on `win_len`

    Args:
        data_wide (pd.DataFrame): Dataset to use
        target_col (str): Target column
        win_len (int): How many columns return back

    Returns:
        List[str]: List of column names

    Examples:
        get_columns_wide_data(gx_volume_w, target_col="y_0", win_len=12)

        ['lag_12', 'lag_11', 'lag_10', 'lag_9', 'lag_8', 'lag_7', 'lag_6',
        'lag_5', 'lag_4', 'lag_3', 'lag_2', 'lag_1']
    """

    # Select range of columns based on `win_len`
    target_idx = data_wide.columns.get_loc(target_col)
    cols_rng = np.arange(target_idx - win_len, target_idx)
    cols_rng = cols_rng[cols_rng >= 0]
    return data_wide.columns[cols_rng].to_list()


def _create_features(volume: pd.DataFrame) -> pd.DataFrame:
    def _growth(s: pd.Series) -> np.ndarray:
        return np.divide(s.iat[-1], s.iat[0]) - 1

    logger.info("Building features")
    df = volume.copy(deep=True)

    # calculate some aggregations
    groups = df.query("month_num <= -1").groupby(["country", "brand"])
    features = groups.agg(
        month_min=("month_num", min),
        growth_before_gx=("volume", _growth),
    )
    return features


def get_dataset_lags(data_wide: pd.DataFrame, target_col: str, win_len: int) -> pd.DataFrame:
    """Return the `win_len` columns prior the target

    Args:
        data_wide (pd.DataFrame): Data in wide format]
        target_col (str): Target column
        win_len (int): How many columns do we want to select

    Returns:
        [pd.DataFrame]: Subset of columns
    """
    # get range of columns from the target backwards
    cols = get_columns_range(data_wide, target_col, win_len)
    return data_wide.loc[:, cols]


def _merge_datasets(metadata, features, volume_wide, months_wide, target_col: str, win_len: int):
    # target month data
    this_month = months_wide.loc[:, target_col].rename("month_name")
    # volume past data
    volume_lags = get_dataset_lags(volume_wide, target_col, win_len=win_len)
    # join everything and create a DataFrame
    y = volume_wide.loc[:, target_col]
    X = (
        metadata.merge(features, how="left", left_index=True, right_index=True)
        .merge(this_month, left_index=True, right_index=True)
        .merge(volume_lags, left_index=True, right_index=True)
        .reindex(y.index)
        .reset_index()
    )
    # Show with how many products we will train our model
    mask = get_products_for_train(volume_wide[target_col])
    logger.info(f"Dataset for column: {target_col} with {X.shape[0]} ({mask.sum()}) products and {X.shape[1]} features")
    return X, y


def create_features_for_models(metadata, volume, target_cols: List[str], win_len: int) -> None:  # is_train: bool
    """Create features for each model and save them in files"""

    # Create wide dataframes
    volume_wide = series_to_supervised(volume, col="volume_normalized")
    months_wide = series_to_supervised(volume, col="month_name")
    # Growth, month_min and other features
    features = _create_features(volume)

    logger.info("Creating training datasets")
    for target_col in target_cols:
        X, y = _merge_datasets(metadata, features, volume_wide, months_wide, target_col=target_col, win_len=win_len)

        sub_folder = f"win_len_{win_len}"
        output_path = PATH_DATA_PROCESSED / sub_folder
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"features_for_{target_col}.joblib"
        logger.info(f"Saving features in : {output_file}")
        dump(dict(X=X, y=y), output_file)


def main():
    """Create features"""
    args = _parse_args()

    # Load dataset
    generics = GenericsDataset()
    volume_df = generics["gx_volume"]
    metadata_df = generics.get_metadata()

    # Normalize volume based on generics entry date
    scaler = VolumeNormalizer(column="volume")
    volume_df["volume_normalized"] = scaler.fit_transform(volume_df)  # pylint: disable=unsupported-assignment-operation

    # # Create features
    # features_df = _create_features(volume_df)
    # logger.info(features_df)

    # # Get features for one target
    # volume_wide = series_to_supervised(volume_df, col="volume_normalized")
    # months_wide = series_to_supervised(volume_df, col="month_name")
    # X, y = _merge_datasets(metadata_df, features_df, volume_wide, months_wide, target_col="y_0", win_len=6)
    # logger.info(X)

    create_features_for_models(metadata_df, volume_df, target_cols=args.target_cols, win_len=args.win_len)

    # # Create features
    # features = BuildFeatures(metadata_df, volume_df)
    # features_df = features._create_features()
    # logger.info(features_df.head())

    # # Get features for one target
    # X, y = features.get_train_data(target_col="y_0", win_len=2)
    # # logger.info(X.head())

    # Create datasets for all the models and store the in data / processed
    # target_cols = [f"y_{i}" for i in range(24)]
    # features.create_features_for_models(target_cols=args.target_cols, win_len=args.win_len)  # is_train=args.is_train


def _parse_args():
    parser = argparse.ArgumentParser()

    all_targets = [f"y_{i}" for i in range(24)]
    parser.add_argument(
        "--target_cols",
        type=str,
        nargs="+",
        default=all_targets,
        help="Target columns that we want to create the feature datasets.",
    )
    parser.add_argument(
        "--win_len",
        type=int,
        default=48,
        help="How many past observations use from the volume use.",
    )

    # Log parameters
    logger.info("Settings used to create the feature datasets:")
    logger_args(logger, parser)

    return parser.parse_args()


if __name__ == "__main__":
    main()
    # _parse_args()

    # Call from terminal
    # python src/features/build_features.py --target_cols y_0 y_1 y_2 --win_len 48
