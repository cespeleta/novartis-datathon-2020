"""Dataset class to be extended by dataset-specific classes."""
from pathlib import Path


class Dataset:
    """Simple abstract class for datasets."""

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "data"

    def load_data(self):
        pass
