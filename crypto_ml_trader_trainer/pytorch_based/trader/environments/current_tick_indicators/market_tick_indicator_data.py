import json
import os
from typing import List

import pandas as pd
import pandas_ta


class MarketTickIndicatorData:

    _cached_data_path = "./cache/MarketTickIndicatorData.json"

    @classmethod
    def prepare_data(cls, data_to_process: pd.DataFrame, index_jump: int) -> List:

        if os.path.isfile(cls._cached_data_path):
            with open(cls._cached_data_path, 'r') as file:
                return json.load(file)

        data = data_to_process.copy()

        macd_period = 5

        data["ma-5min"] = data["close"].rolling(window=5).mean()
        data["ma-30min"] = data["close"].rolling(window=30).mean()
        data["ma-180min"] = data["close"].rolling(window=180).mean()
        data["ma-360min"] = data["close"].rolling(window=360).mean()
        data["rsi-14days"] = data.ta.rsi(length=14 * 60 * 24) / 100
        data["rsi-14h"] = data.ta.rsi(length=14 * 60) / 100
        data["rsi-3h"] = data.ta.rsi(length=3 * 60)
        data["ema-12"] = data.ta.ema(length=12 * macd_period)
        data["ema-26"] = data.ta.ema(length=26 * macd_period)
        data["rvi"] = data.ta.rvi() / 100

        data = data.join(data.ta.bbands(length=120))
        data = data.join(data.ta.stoch(length=60))

        data["macd"] = data["ema-26"] - data["ema-12"]
        for_signal = pd.DataFrame(data["macd"])
        for_signal.rename(inplace=True, columns={"macd": "close"})

        data["macd-signal"] = for_signal.ta.ema(length=9)
        data["macd-hist"] = data["macd"] - data["macd-signal"]
        data = data.dropna()

        # Determine which column to normalize / scale
        scaled_data = []
        cols_to_normalize_template: [str] = ["ma-", "ema-", "close", "high", "low", "open", "BBU", "BBM", "BBL"]
        cols_to_normalize: [str] = []

        for col in [col for col in data.columns for col_to_norm in cols_to_normalize_template if
                    col.startswith(col_to_norm)]:
            cols_to_normalize.append(col)

        for idx in range(0, len(data), index_jump):
            time_idx = data.index[idx]
            row = data.iloc[idx]
            close = row["close"]

            field_to_normalize = row[cols_to_normalize]

            max_value = field_to_normalize.max()
            min_value = field_to_normalize.min()
            max_diff = max(abs(max_value - close), abs(close - min_value))

            for col in cols_to_normalize:
                data.at[time_idx, col] = cls._scale(row[col], close, max_diff)

            scaled_data.append(row.to_list())

            if idx % (100 * index_jump) == 0:
                print(f"{idx}/{len(data)}")

        print("Data prepared")

        with open(cls._cached_data_path, 'w') as file:
            file.write(json.dumps(scaled_data))

        return scaled_data


    @classmethod
    def _scale(cls, number: float, center: float, scale_factor: float) -> float:
        """
         Scale a number, by first centering it and dividing it rescale it.
        :param number:
        :param factor:
        :return:
        """
        return (number - center) / scale_factor