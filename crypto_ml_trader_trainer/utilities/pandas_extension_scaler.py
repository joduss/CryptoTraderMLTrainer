import pandas as pd


@pd.api.extensions.register_series_accessor("scaler")
class FeatureScaler:

    def __init__(self, serie: pd.Series):
        self.serie = serie


    def normalize_01(self, min=None, max=None) -> float:
        """
        Normalize the data between 0 and 1.
        The min value of the original serie will be 0 and the max 1.
        @return:
        """
        min_value = self.serie.min() if min is None else min
        max_value = self.serie.max() if min is None else max

        return (self.serie - min_value) / (max_value - min_value)

    def standardize(self) -> float:
        mean = self.serie.mean()
        std = self.serie.std()
        return (self.serie - self.serie.mean()) / self.serie.std()


@pd.api.extensions.register_dataframe_accessor("scaler")
class FeatureScaler:

    def __init__(self, data_frame: pd.DataFrame):
        self.data_frame = data_frame


    def normalize_01(self, min=None, max=None) -> float:
        """
        Normalize the data between 0 and 1.
        The min value of the original serie will be 0 and the max 1.
        @return:
        """
        min_value = self.data_frame.min() if min is None else min
        max_value = self.data_frame.max() if min is None else max

        return (self.data_frame - min_value) / (max_value - min_value)
