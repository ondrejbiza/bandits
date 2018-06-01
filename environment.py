import numpy as np


class StationaryEnvironment:

  def __init__(self, num_actions, init_mean, init_std, noise_mean, noise_std):
    """
    Initialize an environment for bandits.
    :param num_actions:     Number of actions.
    :param init_mean:       Reward mean.
    :param init_std:        Reward standard deviation.
    :param noise_mean:      Reward noise mean.
    :param noise_std:       Raward noise standard deviation.
    """

    self.num_actions = num_actions
    self.init_mean = init_mean
    self.init_std = init_std
    self.noise_mean = noise_mean
    self.noise_std = noise_std

    self.action_values = None
    self.reset()

  def reset(self):
    """
    Reset the environment.
    :return:    None.
    """

    self.action_values = np.random.normal(self.init_mean, self.noise_std, size=self.num_actions)

  def act(self, action):
    """
    Take an action in the environment.
    :param action:    An action (index from 0 to num_actions - 1).
    :return:          Reward for the action plus noise.
    """

    assert 0 <= action < self.num_actions

    value = self.action_values[action]
    value += np.random.normal(self.noise_mean, self.noise_std)

    return value

class NonStationaryEnvironment:

  def __init__(self, num_actions, init_value, walk_std, noise_mean, noise_std):

    self.num_actions = num_actions
    self.init_value = init_value
    self.walk_std = walk_std
    self.noise_mean = noise_mean
    self.noise_std = noise_std

    self.action_values = None
    self.reset()

  def reset(self):
    """
    Reset the environment.
    :return:    None.
    """

    self.action_values = np.zeros(self.num_actions) + self.init_value

  def act(self, action):
    """
    Take an action in the environment.
    :param action:    An action (index from 0 to num_actions - 1).
    :return:          Reward for the action plus noise.
    """

    assert 0 <= action < self.num_actions

    value = self.action_values[action]
    value += np.random.normal(self.noise_mean, self.noise_std)

    return value

  def step(self):
    """
    Take a single step in a random walk.
    :return:      None.
    """

    self.action_values += np.random.normal(0, self.walk_std, size=self.num_actions)