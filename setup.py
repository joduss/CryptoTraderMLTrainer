from setuptools import setup

setup(
   name='Crypto ML Trader Trainer',
   version='1.2',
   description='A useful module',
   author='Man Foo',
   author_email='foomail@foo.com',
   packages=['crypto_ml_trader_trainer'],
   install_requires=['torch', 'stable_baselines3', 'pandas-ta'],
)