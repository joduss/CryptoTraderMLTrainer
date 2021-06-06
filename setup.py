from setuptools import find_packages, setup

setup(
   name='crypto_ml_trader_trainer',
   version='1.2',
   description='A useful module',
   author='Man Foo',
   author_email='foomail@foo.com',
   packages=find_packages(),
   install_requires=['torch', 'stable_baselines3', 'pandas', 'pandas-ta', 'matplotlib', 'tensorflow'],
)