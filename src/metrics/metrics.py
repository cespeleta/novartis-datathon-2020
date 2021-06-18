from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from src.config import PATH_DATA


def _get_intervals(y_true: np.ndarray, y_pred: np.ndarray, mult: float = 1.96) -> Tuple[np.ndarray, np.ndarray, float]:
    """Confidence intervals using standard deviation of residuals

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        mult (float, optional): Number of standard deviations. For 95% confidence interval use 1.96. Defaults to 1.96.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: Lower confidence interval, High C.I and standard deviation
    """
    std = np.subtract(y_true, y_pred).std() * mult
    return y_pred - std, y_pred + std, std


def coverage_fraction_score(y_true: np.ndarray, y_pred: np.ndarray, mult: float = 1.96) -> float:
    """Coverage fraction score
    We can also evaluate the ability of the two extreme quantile estimators at producing
    a well-calibrated conditational 90%-confidence interval.

    This function is designed to be used within the `cross_validate` function.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        mult (float, optional): Number of standard deviations]. Defaults to 1.96.

    Returns:
        float: Coverage fraction
    """
    y_low, y_high, _ = _get_intervals(y_true, y_pred, mult)
    return coverage_fraction(y_true, y_low, y_high)


def coverage_fraction(y: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> float:
    """Coverage Fraction
    We can also evaluate the ability of the two extreme quantile estimators at producing
    a well-calibrated conditational 90%-confidence interval.

    Args:
        y (np.ndarray): [description]
        y_low (np.ndarray): [description]
        y_high (np.ndarray): [description]

    Returns:
        float: Coverage fraction
    """
    return float(np.mean(np.logical_and(y >= y_low, y <= y_high)))


def show_metrics(cv_results: Dict[str, np.ndarray], scoring: Dict[str, Callable]) -> None:
    """Print metrics in console

    Args:
        cv_results (Dict[str, np.ndarray]): cross validation results
        scoring (Dict[str, Callable]): metrics used in the cv setup
    """
    for k in scoring.keys():
        default = np.array(0)
        print("    ", end="")
        print(
            "Train {:s}: {:.3f} +/- {:.3f} | Validation: {:.3f} +/- {:.3f}".format(
                k,
                cv_results.get(f"train_{k}", default).mean(),
                cv_results.get(f"train_{k}", default).std(),
                cv_results.get(f"test_{k}", default).mean(),
                cv_results.get(f"test_{k}", default).std(),
            )
        )


# def show_metrics2(cv_results: Dict[str, np.ndarray]) -> None:
#     print(
#         "Train score: "
#         f"{cv_results['train_score'].mean():.3f} +/- {cv_results['train_score'].std():.3f}",
#         "| Validation: ",
#         f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}",
#     )


class GenericsCustomMetric:
    """Generics Custom Metric"""

    def __init__(self):
        self.volume_df = pd.read_csv(PATH_DATA / "gx_volume.csv", index_col=0)
        self.id_cols = ["country", "brand"]
        self.avg_vol_df = self._calculate_avg_volume()
        # print(self.avg_vol_df)

        # Calculate volume of previous 12 months

    def _calculate_avg_volume(self) -> pd.Series:
        """Average volume on the 12 months before generics entry

        Returns:
            pd.Series: Pandas series with the average volume for the 12 months before generic entry
        """
        avg_vol = (
            self.volume_df.query("month_num >= -12 and month_num < 0")
            .groupby(self.id_cols)["volume"]
            .mean()
            .reset_index()
        )
        return avg_vol

    @staticmethod
    def _interval_width(y_low: np.ndarray, y_high: np.ndarray) -> float:
        """Wide intervals are penalized

        Args:
            y_low (np.ndarray): Low confidence interval
            y_high (np.ndarray): High confidence interval

        Returns:
            float: Width of the interval
        """
        return np.sum(np.abs(np.subtract(y_low, y_high)))

    @staticmethod
    def _interval_point(y_low: np.ndarray, y_true: np.ndarray, y_high: np.ndarray) -> float:
        """If actuals are outside of the intervals, it adds error

        Args:
            y_low (np.ndarray): Low quantile
            y_true (np.ndarray): True values
            y_high (np.ndarray): High quantile

        Returns:
            float: Penalization value
        """
        below_low = np.sum(np.subtract(y_low, y_true) * (y_true < y_low))
        above_high = np.sum(np.subtract(y_true, y_high) * (y_true > y_high))
        return below_low + above_high

    def uncertainty_accuracy(
        self, y_low: np.ndarray, y_true: np.ndarray, y_high: np.ndarray, avg_vol: float = 1.0
    ) -> float:
        """
        This function aims to compute the Uncertainty Metric for the
        Novartis Datathon, 3rd edition.

        Given the actuals followed by the upper_bound and lower_bound intervals and the
        average volume, it will compute the metric score.

        Keyword parameters:
            actuals (float vector): Real value of Y
            upper_bound (float vector): upper_bound forecast interval (percentile 95)
            lower_bound (float vector): lower_bound forecast interval (percentile 5)
            avg_volume (float): Average monthly volume of the 12 months
                                prior to the generic entry.

        Returns:
            error_metric: Uncertainty Metric score (%)
        """
        # Assert that all the sizes are OK
        assert len(y_low) == 24, "`y_low` should have 24 sorted values"
        assert len(y_true) == 24, "`y_true` should have 24 sorted values"
        assert len(y_high) == 24, "`y_high` should have 24 sorted values"

        # Uncertainty for the first 6 months
        w_ = 0.85 * self._interval_width(y_low[:6], y_high[:6])
        o_ = 0.15 * 2 / 0.05 * self._interval_point(y_low[:6], y_true[:6], y_high[:6])
        u_first_6 = 100 * (w_ + o_) / (6 * avg_vol)

        # Uncertainty for the last 18 months
        w_ = 0.85 * self._interval_width(y_low[6:], y_high[6:])
        o_ = 0.15 * 2 / 0.05 * self._interval_point(y_low[6:], y_true[6:], y_high[6:])
        u_last_18 = 100 * (w_ + o_) / (18 * avg_vol)

        # Add up both parts of the metric
        return 0.6 * u_first_6 + 0.4 * u_last_18

    @staticmethod
    def _custom_mape(y_true: np.ndarray, y_pred: np.ndarray, avg_vol: float) -> float:
        """Custom MAPE with Average volume

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            avg_vol (float): Average volume (12 months) prior to generics entry

        Returns:
            float: Custom MAPE
        """
        num_ = np.sum(np.abs(np.subtract(y_true, y_pred)))
        den_ = len(y_true) * avg_vol
        return np.divide(num_, den_)

    def point_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, avg_vol: float = 1) -> float:
        """Point Accuracy Metric"""
        # Compute the first part of the equation
        custom_mape = self._custom_mape(y_true, y_pred, avg_vol)
        six_month_mape = self._custom_mape(y_true[:6], y_pred[:6], avg_vol)
        twelve_month_mape = self._custom_mape(y_true[6:12], y_pred[6:12], avg_vol)
        last_months_mape = self._custom_mape(y_true[12:], y_pred[12:], avg_vol)
        # Compute the custom metric
        custom_metric = 100 * (0.5 * custom_mape + 0.3 * six_month_mape + 0.1 * (twelve_month_mape + last_months_mape))
        return custom_metric

    def _apply_score(self, x: pd.DataFrame) -> pd.Series:
        """Apply this function to each Country - Brand product."""
        d = {}
        d["custom_metric"] = self.point_accuracy(x["actuals"], x["forecast"], x["volume"].values[0])
        d["uncertainty_metric"] = self.uncertainty_accuracy(
            x["lower_bound"], x["actuals"], x["upper_bound"], x["volume"].values[0]
        )
        return pd.Series(d, index=["custom_metric", "uncertainty_metric"])

    def score(self, preds_df: pd.DataFrame) -> pd.DataFrame:
        """Score Point and Uncertainty metrics

        Args:
            preds_df (pd.DataFrame): Prediction

        Returns:
            pd.DataFrame: Metrics of error

        Examples:
            data_dict = {
                "country": ["country_1"] * 24,
                "brand": ["brand_10"] * 24,
                "actuals": [float(1000)] * 24,
                "forecast": [float(950)] * 24,
                # "avg_vol": [10000] * 24,
                "lower_bound": [800] * 24,
                "upper_bound": [1200] * 24,
                "month_num": list(range(24)),
            }

            id_cols = ["country", "brand"]
            df = pd.DataFrame(data_dict, columns=data_dict.keys())
            print(evaluator.score(df))
        """
        preds_df = preds_df.merge(self.avg_vol_df, on=self.id_cols, how="left")
        df_metrics = preds_df.groupby(self.id_cols).apply(self._apply_score)
        return df_metrics


if __name__ == "__main__":
    evaluator = GenericsCustomMetric()
    # y_true = np.ones(24)
    # y_low, y_high = y_true * 0.99, y_true * 1.01
    # print(evaluator.uncertainty_accuracy(y_low, y_true, y_high))
    # print(evaluator.uncertainty_accuracy(y_true, y_true, y_true))
    # print(evaluator.point_accuracy(y_true, y_true))

    data_dict = {
        "country": ["country_1"] * 24,
        "brand": ["brand_10"] * 24,
        "actuals": [float(1000)] * 24,
        "forecast": [float(950)] * 24,
        # "avg_vol": [10000] * 24,
        "lower_bound": [800] * 24,
        "upper_bound": [1200] * 24,
        "month_num": list(range(24)),
    }

    id_cols = ["country", "brand"]
    df = pd.DataFrame(data_dict, columns=data_dict.keys())
    print(evaluator.score(df))
