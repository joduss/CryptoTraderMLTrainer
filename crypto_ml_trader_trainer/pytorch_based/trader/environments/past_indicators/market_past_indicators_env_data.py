import os
from typing import List

import numpy as np
import pandas as pd
# noinspection PyUnresolvedReferences
import pandas_ta
# noinspection PyUnresolvedReferences
import utilities.pandas_extension_scaler
from utilities.DateUtility import dateparse


class MarketPastIndicatorsEnvData:

    def __init__(self, data_to_process: pd.DataFrame, decision_frequency: int, cache_dir: str, history_periods: List[int] = None):
        """
        Prepare the data in the following way:
        Over the whole data, compute the indicators in the last period of max(history_periods),
        where the indicators are computed for each period.
        Then, keep only the examples every decision_frequency.

        Ex: decision_frequency = 5, history_periods = [10, 30].

        Let's ignore the decision_frequency for now.
        We have 2 indicators, but for each indicator, we have data aggregated by 10 and by 30 minutes.
        If we have 40 minutes of data, we have actually 11 data points.

        Now, let's consider the the decision_frequency which is 5 minutes.
        We have therefore a data point at start + 30 minute, start + 25 and start + 40.


        @param data_to_process: OHLCV data with a 1 minute interval.
        @param decision_frequency: How frequently in minutes
        @param cache_dir:
        @param history_periods: List of periods used to compute market indicators.
        """

        self.history_periods = history_periods if history_periods is not None else [5, 15, 30, 60, 180, 720, 1440, 3 * 1440]

        # keep data here
        # allow to get the close price by index form a cache of close price
        self.original_data = data_to_process[~data_to_process.index.duplicated(keep='first')]
        self.index_jump = decision_frequency
        self.processed_data_dataframe = self._prepare_data(self.original_data, decision_frequency, self.history_periods, cache_dir)
        self.processed_data = self.processed_data_dataframe.to_numpy()

        self.close_prices = self.original_data.loc[self.processed_data_dataframe.index]["close"]
        self.dates = self.processed_data_dataframe.index.to_numpy()


    def __getitem__(self, key) -> np.ndarray:
        return self.processed_data[key]

    def __len__(self):
        return len(self.processed_data)


    @classmethod
    def _prepare_data(cls, data_to_process: pd.DataFrame, index_jump: int, history_periods: List[int], cache_dir: str) -> (pd.DataFrame, np.ndarray):

        # Load data from cache
        cached_data_path = None
        if cache_dir is not None:
            if cache_dir.endswith("/"):
                cache_dir = cache_dir[:-1]

            cached_data_path = f"{cache_dir}/MarketPastIndicatorsData-{index_jump}.csv"

            if os.path.isfile(cached_data_path):
                with open(cached_data_path, 'r') as file:
                    return pd.read_csv(cached_data_path, index_col=0, parse_dates=True, date_parser=dateparse)

        # Create data and save it to cache
        data = data_to_process.copy()

        # Augment data with technical indicators for each period.
        for period in history_periods:
            print(f"Computing data for period {period}")

            data[f"ma-{period}min"] = data["close"].rolling(window=period).mean()
            data[f"rsi-{period}min"] = data.ta.rsi(length=period) / 100
            ema12 = data.ta.ema(length=period)
            ema26 = data.ta.ema(length=round(26/12*period))
            data[f"ema-12-x-{period}"] = ema12
            data[f"ema-26-x-{period}"] = ema26
            data[f"rvi-x-{period}"] = data.ta.rvi(length=period) / 100
            data[f"cci-{period}"] = data.ta.cci(length=period).replace([np.inf, -np.inf], np.nan).interpolate().scaler.normalize_01()
            data[f"willr-{period}"] = data.ta.willr(length=period).scaler.normalize_01()

            bbands = data.ta.bbands(length=period)
            bbands.rename(inplace=True, columns={"BBL_1_2.0": f"BBL_1_2.0-{period}",
                                                                                 "BBM_1_2.0": f"BBM_1_2.0-{period}",
                                                                                 "BBU_1_2.0": f"BBU_1_2.0-{period}",
                                                                                 "BBB_1_2.0": f"BBB_1_2.0-{period}"})

            stoch = (data.ta.stoch(length=period) / 100)
            stoch.rename(inplace=True, columns={"STOCHk_14_3_3": f"STOCHk_14_3_3-{period}",
                                                                                 "STOCHd_14_3_3": f"STOCHd_14_3_3-{period}"})
            data = data.join([bbands, stoch])

            macd = ema12 - ema26
            data[f"macd-line-{period}"] = macd
            for_signal = pd.DataFrame(macd, columns=["close"])

            macd_signal = for_signal.ta.ema(length=9)
            # data[f"macd-signal-{period}"] = macd_signal
            data[f"macd-hist-{period}"] = macd - macd_signal
            data[f"macd-ratio-{period}"] = (macd / macd_signal) - 1

        # The first records of each indicator might be nan.
        data = data.dropna()

        # We keep only data-points depending on decision_frequency.
        indices = range(0, len(data), index_jump)
        filtered_data: pd.DataFrame = pd.DataFrame(data.iloc[indices])

        MarketPastIndicatorsEnvData.normalize(filtered_data)

        # dropping volume and count
        filtered_data = filtered_data.drop(columns="volume")
        filtered_data = filtered_data.drop(columns="trades")


        # Cache the data.
        if cached_data_path is not None:
            filtered_data_copy = filtered_data.copy()
            filtered_data_copy.index = pd.to_datetime(filtered_data_copy.index, unit='s', origin='unix').astype(int) / 10e8
            filtered_data_copy.to_csv(cached_data_path)

        print("Data prepared")
        return filtered_data

    @classmethod
    def normalize(cls, data: pd.DataFrame):
        """
        Normalize the data in-place.
        """

        # Determine which column to normalize / scale
        cols_to_normalize_template: [str] = ["ma-", "ema-", "close", "high", "low", "open", "BBU", "BBM", "BBL", "BBB",
                                             "macd-line"]
        cols_to_normalize: [str] = []

        for col in [col for col in data.columns for col_to_norm in cols_to_normalize_template
                    if col.startswith(col_to_norm)]:
            cols_to_normalize.append(col)

        for idx in range(0, len(data)):
            time_idx = data.index[idx]
            row = data.iloc[idx]
            close = row["close"]

            field_to_normalize = row[cols_to_normalize]

            max_value = field_to_normalize.max()
            min_value = field_to_normalize.min()
            max_diff = max(abs(max_value - close), abs(close - min_value))

            for col in cols_to_normalize:
                data.at[time_idx, col] = cls._scale(row[col], close, max_diff)

            if idx % (100) == 0:
                print(f"{idx}/{len(data)}")


    @classmethod
    def _scale(cls, number: float, center: float, scale_factor: float) -> float:
        """
         Scale a number, by first centering it and dividing it rescale it.
        :param number:
        :param factor:
        :return:
        """
        return (number - center) / scale_factor
