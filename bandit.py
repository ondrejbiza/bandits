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

class UCB1Bandit:

  def __init__(self, env, init=0, alpha=None):
    """
    UCB1 bandit.
    :param env:         Bandit environment.
    :param init:        Initial action values.
    :param alpha:       Alpha for action value sample averages. None means 1 / num_steps.
    """

    self.env = env
    self.init = init
    self.alpha = alpha

    self.action_values = None
    self.action_counts = None
    self.num_plays = None
    self.actions = None
    self.rewards = None

    self.reset()

  def act(self):
    """
    Take a single action in an environment.
    :return:    None.
    """

    # play each arm once
    action = None
    for i, count in enumerate(self.action_counts):

      if count == 0:

        action = i
        break

    # choose action
    if action is None:

      values = np.empty(self.env.num_actions)

      for i in range(self.env.num_actions):
        values[i] = self.action_values[i] + np.sqrt(2 * np.log(self.num_plays) / self.action_counts[i])

      action = np.argmax(values)

    # take an action
    reward = self.env.act(action)

    # update action value
    if self.alpha is None:
      self.action_values[action] += utils.update_mean(reward, self.action_values[action], self.action_counts[action])
    else:
      self.action_values[action] += self.alpha * (reward - self.action_values[action])

    self.action_counts[action] += 1
    self.num_plays += 1

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
    self.num_plays = 0

    self.actions = []
    self.rewards = []

class UCB2Bandit:

  def __init__(self, env, alpha_1, init=0, alpha_2=None):
    """
    UCB2 bandit.
    :param env:         Bandit environment.
    :param alpha_1:     Alpha parameter for UCB2.
    :param init:        Initial action values.
    :param alpha_2:     Alpha for action value sample averages. None means 1 / num_steps.
    """

    self.env = env
    self.alpha_1 = alpha_1
    self.init = init
    self.alpha_2 = alpha_2

    self.action_values = None
    self.action_counts = None
    self.rs = None
    self.times_to_play = None
    self.action_to_play = None
    self.num_plays = None
    self.actions = None
    self.rewards = None

    self.reset()

  def act(self):
    """
    Take a single action in an environment.
    :return:    None.
    """

    # play each arm once
    action = None
    for i, count in enumerate(self.action_counts):
      if count == 0:
        action = i
        break

    # check if there is an action pending
    if self.times_to_play is not None and self.times_to_play != 0:

      action = self.action_to_play
      self.times_to_play -= 1
      assert self.times_to_play >= 0

    # choose action
    if action is None:

      values = np.empty(self.env.num_actions)

      for i in range(self.env.num_actions):
        values[i] = self.action_values[i] + self.__compute_a(self.num_plays, self.rs[i])

      action = int(np.argmax(values))

      self.action_to_play = action

      self.times_to_play = max(self.__compute_tau(self.rs[action] + 1) - self.__compute_tau(self.rs[action]), 1)
      # times to play - 1 because the action will be played once at the end of this function
      self.times_to_play -= 1

      # only add once
      self.rs[action] += 1

    # take an action
    reward = self.env.act(action)

    # update action value
    if self.alpha_2 is None:
      self.action_values[action] += utils.update_mean(reward, self.action_values[action], self.action_counts[action])
    else:
      self.action_values[action] += self.alpha_2 * (reward - self.action_values[action])

    self.action_counts[action] += 1
    self.num_plays += 1

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
    self.rs = np.zeros(self.env.num_actions, dtype=np.int32)
    self.times_to_play = 0
    self.num_plays = 0

    self.actions = []
    self.rewards = []

  def __compute_tau(self, r):

    return int(np.ceil((1.0 + self.alpha_1) ** r))

  def __compute_a(self, n, r):

    term1 = (1 + self.alpha_1) * np.log((np.e * n) / self.__compute_tau(r))
    term2 = 2 * self.__compute_tau(r)

    x =  np.sqrt(term1 / term2)

    if np.isnan(x):
      print("NAN")
      print(n)
      print(term1)
      print(term2)

    return x