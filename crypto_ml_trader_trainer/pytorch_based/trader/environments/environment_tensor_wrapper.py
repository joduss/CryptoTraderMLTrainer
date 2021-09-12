import gym
import torch

from pytorch_based.core.pytorch_global_config import Device
from pytorch_based.trader.environments.market_env_logic import MarketEnvironmentState, MarketStep
from pytorch_based.trader.environments.market_environment import MarketEnvironment


class MarketEnvironmentTensorWrapper(gym.Env):

    def __init__(self, env: MarketEnvironment):
        self.wrapped_env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    #             observation (object): agent's observation of the current environment
    #             reward (float) : amount of reward returned after previous action
    #             done (bool): whether the episode has ended, in which case further step() calls will return undefined results
    #             info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
    def step(self, action: int) -> MarketStep:
        observation, reward, done = self.wrapped_env.step(action)

        observation = MarketEnvironmentState(indicators=torch.from_numpy(observation.indicators).to(Device.device),
                                             wallet=torch.from_numpy(observation.wallet).to(Device.device),
                                             valid_actions_mask=torch.from_numpy(observation.valid_actions_mask).to(Device.device))

        return MarketStep(next_state=observation, reward=reward, ended=done)


    def reset(self) -> MarketEnvironmentState:
        observation: MarketEnvironmentState = self.wrapped_env.reset()

        return MarketEnvironmentState(indicators=torch.from_numpy(observation.indicators).to(Device.device),
                                             wallet=torch.from_numpy(observation.wallet).to(Device.device),
                                             valid_actions_mask=torch.from_numpy(observation.valid_actions_mask).to(Device.device))


    def render(self, mode='human'):
        self.wrapped_env.render(mode)