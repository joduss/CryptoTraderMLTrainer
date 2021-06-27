import json
import os

import numpy as np
import pandas as pd
import pandas_ta
import utilities.pandas_extension_scaler

class MarketTickIndicatorData:

    @classmethod
    def prepare_data(cls, data_to_process: pd.DataFrame, index_jump: int) -> np.array:

        cached_data_path = f"./cache/MarketTickIndicatorData-{index_jump}.json"

        # Loads from cache
        if os.path.isfile(cached_data_path):
            with open(cached_data_path, 'r') as file:
                return np.array(json.load(file))

        # Create data and save it to cache
        data = data_to_process.copy()

        # All periods/lengths are in minutes
        macd_period = 10

        data["ma-5min"] = data["close"].rolling(window=5).mean()
        data["ma-30min"] = data["close"].rolling(window=30).mean()
        data["ma-180min"] = data["close"].rolling(window=180).mean()
        data["ma-360min"] = data["close"].rolling(window=360).mean()
        data["ma-1d"] = data["close"].rolling(window=1440).mean()
        data["ma-1d"] = data["close"].rolling(window=1440).mean()
        data["ma-3d"] = data["close"].rolling(window=3*1440).mean()
        data["rsi-14d"] = data.ta.rsi(length=14 * 60 * 24) / 100
        data["rsi-14h"] = data.ta.rsi(length=14 * 60) / 100
        data["rsi-3h"] = data.ta.rsi(length=3 * 60) / 100
        data["ema-12"] = data.ta.ema(length=12 * macd_period)
        data["ema-26"] = data.ta.ema(length=26 * macd_period)
        data["rvi"] = data.ta.rvi() / 100
        data["cci-1day"] = data.ta.cci(length=1440).scaler.normalize_01()
        data["cci-3days"] = data.ta.cci(length=3*1440).scaler.normalize_01()
        data["cci-6h"] = data.ta.cci(length=360).scaler.normalize_01()
        data["willr-1d"] = data.ta.willr(length=24*60*5).scaler.normalize_01()

        data = data.join(data.ta.bbands(length=120))
        data = data.join(data.ta.stoch(length=60) / 100)
        data = data.drop(columns="BBB_120_2.0")

        data["macd"] = (data["ema-26"] - data["ema-12"])
        for_signal = pd.DataFrame(data["macd"])
        for_signal.rename(inplace=True, columns={"macd": "close"})

        data["macd-signal"] = for_signal.ta.ema(length=9)
        data["macd-hist"] = data["macd"] - data["macd-signal"]
        data = data.dropna()

        # Determine which column to normalize / scale
        # scaled_data = []
        cols_to_normalize_template: [str] = ["ma-", "ema-", "close", "high", "low", "open", "BBU", "BBM", "BBL"]
        cols_to_normalize: [str] = []

        for col in [col for col in data.columns for col_to_norm in cols_to_normalize_template
                    if col.startswith(col_to_norm)]:
            cols_to_normalize.append(col)

        indices = range(0, len(data), index_jump)
        filtered_data = pd.DataFrame(data.iloc[indices])

        for idx in range(0, len(indices)):
            time_idx = filtered_data.index[idx]
            row = filtered_data.iloc[idx]
            close = row["close"]

            field_to_normalize = row[cols_to_normalize]

            max_value = field_to_normalize.max()
            min_value = field_to_normalize.min()
            max_diff = max(abs(max_value - close), abs(close - min_value))

            for col in cols_to_normalize:
                filtered_data.at[time_idx, col] = cls._scale(row[col], close, max_diff)

            # scaled_data.append(row.to_list())

            if idx % (100) == 0:
                print(f"{idx}/{len(filtered_data)}")


        # macd features scaling
        min_value = min(filtered_data["macd"].min(), filtered_data["macd-signal"].min())
        max_value = max(filtered_data["macd"].max(), filtered_data["macd-signal"].max())
        filtered_data["macd"] = filtered_data["macd"].scaler.normalize_01(min=min_value, max=max_value)
        filtered_data["macd-signal"] = filtered_data["macd-signal"].scaler.normalize_01(min=min_value, max=max_value)
        filtered_data["macd-hist"] = filtered_data["macd-hist"].scaler.normalize_01()

        # dropping volume and count
        filtered_data = filtered_data.drop(columns="volume")
        filtered_data = filtered_data.drop(columns="trades")

        print("Data prepared")

        filtered_data = filtered_data.to_numpy()

        with open(cached_data_path, 'w') as file:
            file.write(json.dumps(filtered_data.tolist()))

        return filtered_data


    @classmethod
    def _scale(cls, number: float, center: float, scale_factor: float) -> float:
        """
         Scale a number, by first centering it and dividing it rescale it.
        :param number:
        :param factor:
        :return:
        """
        return (number - center) / scale_factor
