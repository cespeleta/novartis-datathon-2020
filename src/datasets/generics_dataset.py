"""
Generics dataset. Read all provided files and creates a master table.
"""

from pathlib import Path

import pandas as pd

from src.datasets.dataset import Dataset
from src.utils import get_logger

logger = get_logger("generics_dataset")
RAW_DATA_DIRNAME = Dataset.data_dirname()


class GenericsDataset(Dataset):
    """
    Generics Dataset
    """

    def __init__(self):
        self.files = {
            "gx_num_generics": RAW_DATA_DIRNAME / "gx_num_generics.csv",
            "gx_package": RAW_DATA_DIRNAME / "gx_package.csv",
            "gx_panel": RAW_DATA_DIRNAME / "gx_panel.csv",
            "gx_therapeutic_area": RAW_DATA_DIRNAME / "gx_therapeutic_area.csv",
            "gx_volume": RAW_DATA_DIRNAME / "gx_volume.csv",
            "submission_template": RAW_DATA_DIRNAME / "submission_template.csv",
        }
        self.gx_num_generics = None
        self.gx_package = None
        self.gx_panel = None
        self.gx_therapeutic_area = None
        self.gx_volume = None
        self.submission_template = None

        self.load_data()

    def load_data(self) -> None:
        self.gx_num_generics = self._read_file(self.files.get("gx_num_generics"))
        self.gx_package = self._read_file(self.files.get("gx_package"))
        self.gx_panel = self._read_file(self.files.get("gx_panel"))
        self.gx_therapeutic_area = self._read_file(self.files.get("gx_therapeutic_area"))
        self.gx_volume = self._read_file(self.files.get("gx_volume"))
        self.submission_template = self._read_file(self.files.get("submission_template"))

    def __getitem__(self, key: str) -> pd.DataFrame:
        if key == "gx_num_generics":
            return self.gx_num_generics
        if key == "gx_package":
            return self.gx_package
        if key == "gx_panel":
            return self.gx_panel
        if key == "gx_therapeutic_area":
            return self.gx_therapeutic_area
        if key == "gx_volume":
            return self.gx_volume
        if key == "submission_template":
            return self.submission_template

        # Otherwise raise error
        valid_keys = [
            "gx_num_generics",
            "gx_package",
            "gx_panel",
            "gx_therapeutic_area",
            "gx_volume",
            "submission_template",
        ]
        raise ValueError(f"`{key}` is not a valid key. Valid keys are: {valid_keys}")

    def process_panel_data(self) -> pd.DataFrame:
        gx_panel_w = self.gx_panel.pivot_table(
            values="channel_rate", index=["country", "brand"], columns="channel", fill_value=0
        )
        gx_panel_w = gx_panel_w.add_prefix("channel_rate_")
        gx_panel_w.reset_index(inplace=True)
        return gx_panel_w

    def get_metadata(self) -> pd.DataFrame:
        # print("Getting metadata", end=" ")
        gx_metadata = (
            self.gx_volume.loc[:, ["country", "brand"]]
            .drop_duplicates()
            .merge(self.gx_num_generics, how="left", on=["country", "brand"])
            .merge(self.gx_therapeutic_area, how="left", on=["brand"])
            .merge(self.gx_package, how="left", on=["country", "brand"])
            .merge(self.process_panel_data(), how="left", on=["country", "brand"])
            .set_index(["country", "brand"])
        )
        # print(gx_metadata.shape)
        logger.info(f"Getting metadata {gx_metadata.shape}")
        return gx_metadata

    # def create_grid(self) -> pd.DataFrame:
    #     # for each group create a sequence with all the `month_num`.
    #     # `month_num` will go from the init value up to 23
    #     cols = ["country", "brand"]
    #     grid_df = self.gx_volume.groupby(cols).apply(
    #         lambda x: pd.DataFrame({"month_num": np.arange(horizon, min(x.month_num) - 1, -1)[::-1]})
    #     )
    #     grid_df = grid_df.droplevel(level=2, axis=0)
    #     grid_df = grid_df.reset_index()
    #     # append volume and month_name
    #     cols = ["country", "brand", "month_num"]
    #     grid_df = grid_df.merge(self.gx_volume, on=cols, how="left")
    #     grid_df = grid_df.sort_values(by=cols)
    #     # fill missing values on month_name repeeting the existing sequence
    #     cols = ["country", "brand"]
    #     grid_df["month_name"] = grid_df.groupby(cols)["month_name"].transform(lambda x: np.resize(x[:13], x.shape[0]))
    #     return grid_df

    @staticmethod
    def _read_file(file: Path) -> pd.DataFrame:
        # print(f"Loading {file.name}", end=" ")
        df = pd.read_csv(file)
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)
        logger.info(f"Loading {file.name} {df.shape}")
        # print(df.shape)
        return df


def main():
    """Load Generics dataset and print info."""
    dataset = GenericsDataset()

    logger.info(RAW_DATA_DIRNAME)
    logger.info(dataset["gx_num_generics"].head())
    logger.info(dataset.get_metadata().head())


if __name__ == "__main__":
    main()
