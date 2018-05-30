import numpy as np
import utils


class EpsilonGreedyBandit:

  def __init__(self, env, epsilon):

    self.env = env
    self.epsilon = epsilon

    self.action_values = None
    self.action_counts = None
    self.actions = None
    self.rewards = None

    self.reset()

  def act(self):

    # select an action
    r = np.random.uniform(0, 1)
    if r > self.epsilon:
      action = np.argmax(self.action_values)
    else:
      action = np.random.randint(0, self.env.num_actions)

    # take an action
    reward = self.env.act(action)

    # update action value
    self.action_values[action] += utils.update_mean(reward, self.action_values[action], self.action_counts[action])
    self.action_counts[action] += 1

    # save action and reward
    self.actions.append(action)
    self.rewards.append(reward)

  def reset(self):

    self.action_values = np.zeros(self.env.num_actions, dtype=np.float32)
    self.action_counts = np.zeros(self.env.num_actions, dtype=np.int32)

    self.actions = []
    self.rewards = []