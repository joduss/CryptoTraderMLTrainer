import gym
import torch

from ..core.pytorch_global_config import Device


class EnvironmentTensorWrapper(gym.Env):

    def __init__(self, env: gym.Env):
        self.wrapped_env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    #             observation (object): agent's observation of the current environment
    #             reward (float) : amount of reward returned after previous action
    #             done (bool): whether the episode has ended, in which case further step() calls will return undefined results
    #             info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
    def step(self, action):
        observation, reward, done, info = self.wrapped_env.step(action)
        observation = torch.from_numpy(observation).to(Device.device)
        return observation, reward, done, info

    def reset(self):
        return torch.from_numpy(self.wrapped_env.reset()).to(Device.device)

    def render(self, mode='human'):
        self.wrapped_env.render(mode)