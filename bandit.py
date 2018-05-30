import numpy as np
import utils


class GreedyBandit:

  def __init__(self, env):

    self.env = env
    self.action_values = np.zeros(self.env.num_actions, dtype=np.float32)
    self.action_counts = np.zeros(self.env.num_actions, dtype=np.int32)

    self.actions = []
    self.rewards = []


  def act(self):

    # select an action
    action = np.argmax(self.action_values)

    # take an action
    reward = self.env.act(action)

    # update action value
    self.action_values[action] = utils.update_mean(reward, self.action_values[action], self.action_counts[action])
    self.action_counts[action] += 1

    # save action and reward
    self.actions.append(action)
    self.rewards.append(reward)