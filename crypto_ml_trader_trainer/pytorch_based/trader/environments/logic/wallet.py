

class Wallet:

    def update_coin_price(self, new_price: float):
        raise NotImplementedError()

    def coins_value(self) -> float:
        raise NotImplementedError()

    def bought_coins(self, coins: float, price: float):
        raise NotImplementedError()


    def sold_coins(self, price: float, value: float) -> float:
        raise NotImplementedError()


    def net_worth_diff(self, price_now: float, price_before: float = None) -> float:
        raise NotImplementedError()


    def profits(self):
        raise NotImplementedError()


    def can_buy(self) -> bool:
        raise NotImplementedError()

    def can_sell(self) -> bool:
        raise NotImplementedError()
