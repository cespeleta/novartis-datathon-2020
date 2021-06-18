import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import seaborn as sns

from src.config import PATH_DATA

sns.set_style("darkgrid")  # or white
sns.set_context("notebook")  # or paper

# Load data
gx_volume = pd.read_csv(PATH_DATA / "gx_volume.csv", index_col=0)
gx_num_generics = pd.read_csv(PATH_DATA / "gx_num_generics.csv", index_col=0)
gx_therapeutic_area = pd.read_csv(PATH_DATA / "gx_therapeutic_area.csv", index_col=0)
gx_package = pd.read_csv(PATH_DATA / "gx_package.csv", index_col=0)


# https://stackoverflow.com/questions/40566413/matplotlib-pyplot-auto-adjust-unit-of-y-axis
# https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html
def format_func(value, tick_number):  # pylint: disable=unused-argument
    if value >= 1e6:
        return "{:d}M".format(int(value / 1e6))
    if value >= 1e3:
        return "{:d}k".format(int(value / 1e3))
    return "{:.2f}".format(value)


def exponential(x, m, t, b):
    return m * np.exp(-t * x) + b


def fit_exponential(ys):
    # sequence of values
    xs = np.arange(len(ys))

    # perform the fit
    p0 = (max(ys), 0.3, max(ys))  # start with values near those we expect
    params, _ = scipy.optimize.curve_fit(  # pylint: disable=unbalanced-tuple-unpacking
        exponential, xs, ys, p0, maxfev=10_000
    )
    # m, t, b = params
    return params


def add_exp_decay(ys, ax=None):
    # Fit exponential curve and find the params
    m, t, b = fit_exponential(ys)

    # extrapolate to all points
    xs2 = np.arange(24)
    ys2 = exponential(xs2, m, t, b)

    # plot the results
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 5))

    # plot the fitted curve
    _ = ax.plot(xs2, ys2, "--", label="fitted")

    # prepare text formula
    text = f"y = {format_func(m, None)}$e^{{-{t:.2f}x}}$ + {format_func(b, None)}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.2)

    # place a text box in upper left in axes coords
    _ = ax.text(0.02, 0.95, text, fontsize=11, transform=ax.transAxes, ha="left", verticalalignment="top", bbox=props)

    return ax


def _add_predictions(predictions, ax=None):
    _ = ax.plot(predictions["month_num"], predictions["forecast"])
    _ = ax.fill_between(predictions["month_num"], predictions["pred_95_low"], predictions["pred_95_high"], alpha=0.1)
    return ax


def plot_volume(
    cc: str = "country_1",
    bb: str = "brand_3",
    y_col: str = "volume",
    predictions: pd.DataFrame = None,
    add_decay: bool = False,
    ax: plt.Axes = None,
) -> plt.Axes:

    # Subset data
    plot_data = gx_volume.query("country == @@cc and brand == @bb")

    # Time series plot
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax = sns.lineplot(x="month_num", y=y_col, data=plot_data, ax=ax)
    _ = ax.axvline(x=0, ymin=0, ymax=1, ls="-", lw=4, c="k", alpha=0.1)
    _ = ax.set_title(", ".join([cc, bb]))
    _ = ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

    # Color, Alpha and Size
    c, a, s = "brown", 0.8, 10

    # Add extra information
    value = gx_num_generics.query("country == @cc & brand == @bb")["num_generics"].values[0]
    _ = ax.scatter([], [], c=c, alpha=a, s=s, label=f"Num generics: {value}")

    value = gx_therapeutic_area.query("brand == @bb")["therapeutic_area"].values[0]
    _ = ax.scatter([], [], c=c, alpha=a, s=s, label=value)

    value = gx_package.query("country == @cc & brand == @bb")["presentation"].values[0]
    _ = ax.scatter([], [], c=c, alpha=a, s=s, label=f"Presentation: {value}")

    # Show legend
    _ = ax.legend(scatterpoints=1, frameon=True, loc="lower left", fontsize="small", handletextpad=0)

    # Add predictions to the plot
    if predictions is not None:
        this_predictions = predictions.query("country == @@cc and brand == @bb")
        _ = _add_predictions(this_predictions, ax=ax)

    if add_decay:
        _ = add_exp_decay(ys=plot_data.query("month_num >= 0")[y_col].values, ax=ax)

    return ax
