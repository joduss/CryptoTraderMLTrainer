

class WalletInterface:

    def __init__(self):
        self.max_worth = 0
        self.net_worth = 0
        self.balance = 0.0

    def coins_value(self) -> float:
        raise NotImplementedError()

    def profits(self):
        """
        Total profits since the start.
        """
        raise NotImplementedError()

    def bought_coins(self, coins: float, price: float):
        raise NotImplementedError()

    def sold_coins(self, price: float, value: float) -> float:
        raise NotImplementedError()

    def update_coin_price(self, new_price: float):
        raise NotImplementedError()


    def can_buy(self) -> bool:
        raise NotImplementedError()

    def can_sell(self) -> bool:
        raise NotImplementedError()


    def _update_net_worth(self, value: float):
        self.net_worth = value

        if value > self.max_worth:
            self.max_worth = value
