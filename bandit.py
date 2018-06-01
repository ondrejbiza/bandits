import numpy as np
import utils


class EpsilonGreedyBandit:

  def __init__(self, env, epsilon, init=0, alpha=None):
    """
    Epsilon-greedy bandit.
    :param env:         Bandit environment.
    :param epsilon:     Epsilon (probability of choosing a random action).
    :param init:        Initial action values.
    :param alpha:       Alpha for action value sample averages. None means 1 / num_steps.
    """

    self.env = env
    self.epsilon = epsilon
    self.init = init
    self.alpha = alpha

    self.action_values = None
    self.action_counts = None
    self.actions = None
    self.rewards = None

    self.reset()

  def act(self):
    """
    Take a single action in an environment.
    :return:    None.
    """

    # select an action
    r = np.random.uniform(0, 1)
    if r > self.epsilon:
      action = np.argmax(self.action_values)
    else:
      action = np.random.randint(0, self.env.num_actions)

    # take an action
    reward = self.env.act(action)

    # update action value
    if self.alpha is None:
      self.action_values[action] += utils.update_mean(reward, self.action_values[action], self.action_counts[action])
    else:
      self.action_values[action] += self.alpha * (reward - self.action_values[action])

    self.action_counts[action] += 1

    # save action and reward
    self.actions.append(action)
    self.rewards.append(reward)

  def reset(self):
    """
    Reset the bandit.
    :return:    None.
    """

    self.action_values = np.zeros(self.env.num_actions, dtype=np.float32) + self.init
    self.action_counts = np.zeros(self.env.num_actions, dtype=np.int32)

    self.actions = []
    self.rewards = []

class SoftmaxBandit:

  def __init__(self, env, temperature, init=0, alpha=None):
    """
    Epsilon-greedy bandit.
    :param env:             Bandit environment.
    :param temperature:     Temperate for the softmax (the higher the more random actions).
    :param init:            Initial action values.
    :param alpha:           Alpha for action value sample averages. None means 1 / num_steps.
    """

    self.env = env
    self.temperature = temperature
    self.init = init
    self.alpha = alpha

    self.action_values = None
    self.action_counts = None
    self.actions = None
    self.rewards = None

    self.reset()

  def act(self):
    """
    Take a single action in an environment.
    :return:    None.
    """

    # select an action
    softmax = self.__softmax(self.action_values, self.temperature)
    action = np.random.choice(range(self.env.num_actions), p=softmax)

    # take an action
    reward = self.env.act(action)

    # update action value
    if self.alpha is None:
      self.action_values[action] += utils.update_mean(reward, self.action_values[action], self.action_counts[action])
    else:
      self.action_values[action] += self.alpha * (reward - self.action_values[action])

    self.action_counts[action] += 1

    # save action and reward
    self.actions.append(action)
    self.rewards.append(reward)

  def reset(self):
    """
    Reset the bandit.
    :return:    None.
    """

    self.action_values = np.zeros(self.env.num_actions, dtype=np.float32) + self.init
    self.action_counts = np.zeros(self.env.num_actions, dtype=np.int32)

    self.actions = []
    self.rewards = []

  @staticmethod
  def __softmax(x, t):
    """
    Softmax function.
    :param x:   Array of values.
    :param t:   Temperature.
    :return:    Softmax distribution.
    """

    e_x = np.exp(x / t)
    return e_x / e_x.sum()