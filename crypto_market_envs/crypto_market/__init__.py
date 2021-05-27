import gym

# delete if it's registered
env_name = 'crypto-market-indicators-v1'

if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]

gym.envs.register(id=env_name,entry_point='crypto_market.envs:CryptoMarketIndicatorsEnvironment',)

