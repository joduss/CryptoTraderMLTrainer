from enum import Enum

import numpy as np


class TradingAction(Enum):
    BUY = 0
    HOLD = 1
    SELL = 2

    @staticmethod
    def hot_encode(actions) -> np.ndarray:
        valid_actions_hot_encoded = [0, 0, 0]

        for action in actions:
            valid_actions_hot_encoded[action.value] = 1

        return np.array(valid_actions_hot_encoded)
