

class SingleOrderWallet:

    def __init__(self, balance: float):
        self.initial_balance: float = balance
        self.balance: float = balance
        self.coins_balance: float = 0
        self.coin_price: float = 0
        self.initial_coin_price: float = 0
        self.net_worth: float = balance
        self.max_worth: float = 0

        self.transactions: int = 0

    def update_coin_price(self, new_price: float):
        self.coin_price = new_price
        self.net_worth = self.balance + self.coins_value()
        if self.net_worth > self.max_worth:
            self.max_worth = self.net_worth

    def coins_value(self) -> float:
        return self.coin_price * self.coins_balance

    def bought_coins(self, coins: float, price: float):
        self.balance = 0
        self.coins_balance = coins
        self.initial_coin_price = price
        self.update_coin_price(price)
        self.transactions += 1

    def sold_coins(self, price: float, value: float) -> float:
        profits = self.coins_balance * (price - self.initial_coin_price)
        self.balance = value
        self.coins_balance = 0
        self.update_coin_price(price)
        self.transactions += 1

        return profits

    def net_worth_diff(self, price_now: float, price_before: float = None) -> float:
        if price_before is None:
            price_before = self.initial_coin_price

        return self.coins_balance * (price_now - price_before)

    def profits(self):
        return self.balance + self.coins_balance * self.coin_price - self.initial_balance


    def can_buy(self) -> bool:
        return self.balance > 0 and self.coins_balance == 0

    def can_sell(self) -> bool:
        return self.coins_balance > 0
